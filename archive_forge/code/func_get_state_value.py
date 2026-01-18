from typing import List
import gymnasium as gym
from ray.rllib.models.tf.layers import NoisyLayer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def get_state_value(self, model_out: TensorType) -> TensorType:
    """Returns the state value prediction for the given state embedding."""
    return self.state_value_head(model_out)