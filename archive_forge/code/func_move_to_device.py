import inspect
import warnings
import torch
from .state import AcceleratorState, GradientState
from .utils import DistributedType, honor_type, is_torch_xla_available
def move_to_device(state, device):
    if isinstance(state, (list, tuple)):
        return honor_type(state, (move_to_device(t, device) for t in state))
    elif isinstance(state, dict):
        return type(state)({k: move_to_device(v, device) for k, v in state.items()})
    elif isinstance(state, torch.Tensor):
        return state.to(device)
    return state