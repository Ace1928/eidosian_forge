import inspect
import warnings
import torch
from .state import AcceleratorState, GradientState
from .utils import DistributedType, honor_type, is_torch_xla_available
def patch_optimizer_step(accelerated_optimizer: AcceleratedOptimizer, method):

    def patched_step(*args, **kwargs):
        accelerated_optimizer._accelerate_step_called = True
        return method(*args, **kwargs)
    return patched_step