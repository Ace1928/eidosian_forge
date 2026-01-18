from typing import List
import gymnasium as gym
from ray.rllib.models.tf.layers import NoisyLayer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def build_state_score(prefix: str, model_out: TensorType) -> TensorType:
    state_out = model_out
    for i in range(len(q_hiddens)):
        if use_noisy:
            state_out = NoisyLayer('{}dueling_hidden_{}'.format(prefix, i), q_hiddens[i], sigma0)(state_out)
        else:
            state_out = tf.keras.layers.Dense(units=q_hiddens[i], activation=tf.nn.relu)(state_out)
            if add_layer_norm:
                state_out = tf.keras.layers.LayerNormalization()(state_out)
    if use_noisy:
        state_score = NoisyLayer('{}dueling_output'.format(prefix), num_atoms, sigma0, activation=None)(state_out)
    else:
        state_score = tf.keras.layers.Dense(units=num_atoms, activation=None)(state_out)
    return state_score