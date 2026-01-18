import re
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.dreamerv3.tf.models.disagree_networks import DisagreeNetworks
from ray.rllib.algorithms.dreamerv3.tf.models.actor_network import ActorNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.critic_network import CriticNetwork
from ray.rllib.algorithms.dreamerv3.tf.models.world_model import WorldModel
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import inverse_symlog
def dream_trajectory(self, start_states, start_is_terminated):
    """Dreams trajectories of length H from batch of h- and z-states.

        Note that incoming data will have the shapes (BxT, ...), where the original
        batch- and time-dimensions are already folded together. Beginning from this
        new batch dim (BxT), we will unroll `timesteps_H` timesteps in a time-major
        fashion, such that the dreamed data will have shape (H, BxT, ...).

        Args:
            start_states: Dict of `h` and `z` states in the shape of (B, ...) and
                (B, num_categoricals, num_classes), respectively, as
                computed by a train forward pass. From each individual h-/z-state pair
                in the given batch, we will branch off a dreamed trajectory of len
                `timesteps_H`.
            start_is_terminated: Float flags of shape (B,) indicating whether the
                first timesteps of each batch row is already a terminated timestep
                (given by the actual environment).
        """
    a_dreamed_t0_to_H = []
    a_dreamed_dist_params_t0_to_H = []
    h = start_states['h']
    z = start_states['z']
    h_states_t0_to_H = [h]
    z_states_prior_t0_to_H = [z]
    a, a_dist_params = self.actor(h=tf.stop_gradient(h), z=tf.stop_gradient(z))
    a_dreamed_t0_to_H.append(a)
    a_dreamed_dist_params_t0_to_H.append(a_dist_params)
    for i in range(self.horizon):
        h = self.world_model.sequence_model(a=a, h=h, z=z)
        h_states_t0_to_H.append(h)
        z, _ = self.world_model.dynamics_predictor(h=h)
        z_states_prior_t0_to_H.append(z)
        a, a_dist_params = self.actor(h=tf.stop_gradient(h), z=tf.stop_gradient(z))
        a_dreamed_t0_to_H.append(a)
        a_dreamed_dist_params_t0_to_H.append(a_dist_params)
    h_states_H_B = tf.stack(h_states_t0_to_H, axis=0)
    h_states_HxB = tf.reshape(h_states_H_B, [-1] + h_states_H_B.shape.as_list()[2:])
    z_states_prior_H_B = tf.stack(z_states_prior_t0_to_H, axis=0)
    z_states_prior_HxB = tf.reshape(z_states_prior_H_B, [-1] + z_states_prior_H_B.shape.as_list()[2:])
    a_dreamed_H_B = tf.stack(a_dreamed_t0_to_H, axis=0)
    a_dreamed_dist_params_H_B = tf.stack(a_dreamed_dist_params_t0_to_H, axis=0)
    r_dreamed_HxB, _ = self.world_model.reward_predictor(h=h_states_HxB, z=z_states_prior_HxB)
    r_dreamed_H_B = tf.reshape(inverse_symlog(r_dreamed_HxB), shape=[self.horizon + 1, -1])
    if self.use_curiosity:
        results_HxB = self.disagree_nets.compute_intrinsic_rewards(h=h_states_HxB, z=z_states_prior_HxB, a=tf.reshape(a_dreamed_H_B, [-1] + a_dreamed_H_B.shape.as_list()[2:]))
        r_intrinsic_H_B = tf.reshape(results_HxB['rewards_intrinsic'], shape=[self.horizon + 1, -1])[1:]
        curiosity_forward_train_outs = results_HxB['forward_train_outs']
        del results_HxB
    c_dreamed_HxB, _ = self.world_model.continue_predictor(h=h_states_HxB, z=z_states_prior_HxB)
    c_dreamed_H_B = tf.reshape(c_dreamed_HxB, [self.horizon + 1, -1])
    c_dreamed_H_B = tf.concat([1.0 - tf.expand_dims(tf.cast(start_is_terminated, tf.float32), 0), c_dreamed_H_B[1:]], axis=0)
    dream_loss_weights_H_B = tf.math.cumprod(self.gamma * c_dreamed_H_B, axis=0) / self.gamma
    v, v_symlog_dreamed_logits_HxB = self.critic(h=h_states_HxB, z=z_states_prior_HxB, use_ema=False)
    v_dreamed_HxB = inverse_symlog(v)
    v_dreamed_H_B = tf.reshape(v_dreamed_HxB, shape=[self.horizon + 1, -1])
    v_symlog_dreamed_ema_HxB, _ = self.critic(h=h_states_HxB, z=z_states_prior_HxB, use_ema=True)
    v_symlog_dreamed_ema_H_B = tf.reshape(v_symlog_dreamed_ema_HxB, shape=[self.horizon + 1, -1])
    ret = {'h_states_t0_to_H_BxT': h_states_H_B, 'z_states_prior_t0_to_H_BxT': z_states_prior_H_B, 'rewards_dreamed_t0_to_H_BxT': r_dreamed_H_B, 'continues_dreamed_t0_to_H_BxT': c_dreamed_H_B, 'actions_dreamed_t0_to_H_BxT': a_dreamed_H_B, 'actions_dreamed_dist_params_t0_to_H_BxT': a_dreamed_dist_params_H_B, 'values_dreamed_t0_to_H_BxT': v_dreamed_H_B, 'values_symlog_dreamed_logits_t0_to_HxBxT': v_symlog_dreamed_logits_HxB, 'v_symlog_dreamed_ema_t0_to_H_BxT': v_symlog_dreamed_ema_H_B, 'dream_loss_weights_t0_to_H_BxT': dream_loss_weights_H_B}
    if self.use_curiosity:
        ret['rewards_intrinsic_t1_to_H_B'] = r_intrinsic_H_B
        ret.update(curiosity_forward_train_outs)
    if isinstance(self.action_space, gym.spaces.Discrete):
        ret['actions_ints_dreamed_t0_to_H_B'] = tf.argmax(a_dreamed_H_B, axis=-1)
    return ret