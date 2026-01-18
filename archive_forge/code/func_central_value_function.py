from gymnasium.spaces import Box
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
def central_value_function(self, obs, opponent_obs, opponent_actions):
    input_ = torch.cat([obs, opponent_obs, torch.nn.functional.one_hot(opponent_actions.long(), 2).float()], 1)
    return torch.reshape(self.central_vf(input_), [-1])