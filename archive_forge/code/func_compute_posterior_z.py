from typing import Optional
import gymnasium as gym
import tree  # pip install dm_tree
from ray.rllib.algorithms.dreamerv3.tf.models.components.continue_predictor import (
from ray.rllib.algorithms.dreamerv3.tf.models.components.dynamics_predictor import (
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.representation_layer import (
from ray.rllib.algorithms.dreamerv3.tf.models.components.reward_predictor import (
from ray.rllib.algorithms.dreamerv3.tf.models.components.sequence_model import (
from ray.rllib.algorithms.dreamerv3.utils import get_gru_units
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import symlog
def compute_posterior_z(self, observations, initial_h):
    if self.symlog_obs:
        observations = symlog(observations)
    encoder_out = self.encoder(observations)
    posterior_mlp_input = tf.concat([encoder_out, initial_h], axis=-1)
    repr_input = self.posterior_mlp(posterior_mlp_input)
    z_t, _ = self.posterior_representation_layer(repr_input)
    return z_t