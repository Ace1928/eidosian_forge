from collections import OrderedDict
import gymnasium as gym
from typing import Union, Dict, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def pack_state(self, ctrl_hidden: List[Tuple[TensorType, TensorType]], memory_dict: Dict[str, TensorType], read_vecs: TensorType) -> List[TensorType]:
    """Given the dnc output, pack it into a list of tensors
        for rllib state. Order is ctrl_hidden, read_vecs, memory_dict"""
    state = []
    ctrl_hidden = [ctrl_hidden[0][0].permute(1, 0, 2), ctrl_hidden[0][1].permute(1, 0, 2)]
    state += ctrl_hidden
    assert len(state) == 2, 'Failed to verify packed state'
    state.append(read_vecs)
    assert len(state) == 3, 'Failed to verify packed state'
    state += memory_dict.values()
    assert len(state) == 9, 'Failed to verify packed state'
    return state