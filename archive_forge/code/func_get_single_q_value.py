from gymnasium.spaces import Box
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import (
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
def get_single_q_value(self, underlying_output, action):
    input_ = torch.cat([underlying_output, action], dim=-1)
    input_dict = {'obs': input_}
    q_values, _ = self.q_head(input_dict)
    return q_values