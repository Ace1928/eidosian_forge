import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_utils import huber_loss, make_tf_callable
from ray.rllib.utils.typing import (
def sac_actor_critic_loss(policy: Policy, model: ModelV2, dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for the Soft Actor Critic.

    Args:
        policy: The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch: The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    deterministic = policy.config['_deterministic_loss']
    _is_training = policy._get_is_training_placeholder()
    model_out_t, _ = model(SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=_is_training), [], None)
    model_out_tp1, _ = model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=_is_training), [], None)
    target_model_out_tp1, _ = policy.target_model(SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=_is_training), [], None)
    if model.discrete:
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        log_pis_t = tf.nn.log_softmax(action_dist_inputs_t, -1)
        policy_t = tf.math.exp(log_pis_t)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        log_pis_tp1 = tf.nn.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = tf.math.exp(log_pis_tp1)
        q_t, _ = model.get_q_values(model_out_t)
        q_tp1, _ = policy.target_model.get_q_values(target_model_out_tp1)
        if policy.config['twin_q']:
            twin_q_t, _ = model.get_twin_q_values(model_out_t)
            twin_q_tp1, _ = policy.target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_tp1 -= model.alpha * log_pis_tp1
        one_hot = tf.one_hot(train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1])
        q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
        if policy.config['twin_q']:
            twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
        q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
        q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32)) * q_tp1_best
    else:
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        action_dist_t = action_dist_class(action_dist_inputs_t, policy.model)
        policy_t = action_dist_t.sample() if not deterministic else action_dist_t.deterministic_sample()
        log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, policy.model)
        policy_tp1 = action_dist_tp1.sample() if not deterministic else action_dist_tp1.deterministic_sample()
        log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)
        q_t, _ = model.get_q_values(model_out_t, tf.cast(train_batch[SampleBatch.ACTIONS], tf.float32))
        if policy.config['twin_q']:
            twin_q_t, _ = model.get_twin_q_values(model_out_t, tf.cast(train_batch[SampleBatch.ACTIONS], tf.float32))
        q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
        if policy.config['twin_q']:
            twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = tf.reduce_min((q_t_det_policy, twin_q_t_det_policy), axis=0)
        q_tp1, _ = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config['twin_q']:
            twin_q_tp1, _ = policy.target_model.get_twin_q_values(target_model_out_tp1, policy_tp1)
            q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
        q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
        if policy.config['twin_q']:
            twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        q_tp1 -= model.alpha * log_pis_tp1
        q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (1.0 - tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32)) * q_tp1_best
    q_t_selected_target = tf.stop_gradient(tf.cast(train_batch[SampleBatch.REWARDS], tf.float32) + policy.config['gamma'] ** policy.config['n_step'] * q_tp1_best_masked)
    base_td_error = tf.math.abs(q_t_selected - q_t_selected_target)
    if policy.config['twin_q']:
        twin_td_error = tf.math.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error
    prio_weights = tf.cast(train_batch[PRIO_WEIGHTS], tf.float32)
    critic_loss = [tf.reduce_mean(prio_weights * huber_loss(base_td_error))]
    if policy.config['twin_q']:
        critic_loss.append(tf.reduce_mean(prio_weights * huber_loss(twin_td_error)))
    if model.discrete:
        alpha_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.stop_gradient(policy_t), -model.log_alpha * tf.stop_gradient(log_pis_t + model.target_entropy)), axis=-1))
        actor_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(policy_t, model.alpha * log_pis_t - tf.stop_gradient(q_t)), axis=-1))
    else:
        alpha_loss = -tf.reduce_mean(model.log_alpha * tf.stop_gradient(log_pis_t + model.target_entropy))
        actor_loss = tf.reduce_mean(model.alpha * log_pis_t - q_t_det_policy)
    policy.policy_t = policy_t
    policy.q_t = q_t
    policy.td_error = td_error
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.alpha_loss = alpha_loss
    policy.alpha_value = model.alpha
    policy.target_entropy = model.target_entropy
    return actor_loss + tf.math.add_n(critic_loss) + alpha_loss