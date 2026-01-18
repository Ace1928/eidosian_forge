from ray.rllib.models.tf.tf_action_dist import Categorical, ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
@staticmethod
def required_model_output_shape(action_space, model_config):
    return 16