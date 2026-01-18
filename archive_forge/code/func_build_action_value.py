from typing import List
import gymnasium as gym
from ray.rllib.models.tf.layers import NoisyLayer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def build_action_value(prefix: str, model_out: TensorType) -> List[TensorType]:
    if q_hiddens:
        action_out = model_out
        for i in range(len(q_hiddens)):
            if use_noisy:
                action_out = NoisyLayer('{}hidden_{}'.format(prefix, i), q_hiddens[i], sigma0)(action_out)
            elif add_layer_norm:
                action_out = tf.keras.layers.Dense(units=q_hiddens[i], activation=tf.nn.relu)(action_out)
                action_out = tf.keras.layers.LayerNormalization()(action_out)
            else:
                action_out = tf.keras.layers.Dense(units=q_hiddens[i], activation=tf.nn.relu, name='hidden_%d' % i)(action_out)
    else:
        action_out = model_out
    if use_noisy:
        action_scores = NoisyLayer('{}output'.format(prefix), self.action_space.n * num_atoms, sigma0, activation=None)(action_out)
    elif q_hiddens:
        action_scores = tf.keras.layers.Dense(units=self.action_space.n * num_atoms, activation=None)(action_out)
    else:
        action_scores = model_out
    if num_atoms > 1:
        z = tf.range(num_atoms, dtype=tf.float32)
        z = v_min + z * (v_max - v_min) / float(num_atoms - 1)

        def _layer(x):
            support_logits_per_action = tf.reshape(tensor=x, shape=(-1, self.action_space.n, num_atoms))
            support_prob_per_action = tf.nn.softmax(logits=support_logits_per_action)
            x = tf.reduce_sum(input_tensor=z * support_prob_per_action, axis=-1)
            logits = support_logits_per_action
            dist = support_prob_per_action
            return [x, z, support_logits_per_action, logits, dist]
        return tf.keras.layers.Lambda(_layer)(action_scores)
    else:
        logits = tf.expand_dims(tf.ones_like(action_scores), -1)
        dist = tf.expand_dims(tf.ones_like(action_scores), -1)
        return [action_scores, logits, dist]