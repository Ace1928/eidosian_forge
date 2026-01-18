from gymnasium.spaces import Box
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import (
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch
def get_q_values(self, underlying_output):
    v = self.V(underlying_output)
    a = self.A(underlying_output)
    advantages_mean = torch.mean(a, 1)
    advantages_centered = a - torch.unsqueeze(advantages_mean, 1)
    return v + advantages_centered