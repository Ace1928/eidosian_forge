import functools
import weakref
import torch.nn
from torch.nn import Module
from .utils import ExactWeakKeyDictionary, is_lazy_module
def is_dynamic_nn_module(obj):
    """Check for nn.Modules() created dynamically or mutated"""
    if isinstance(obj, torch.nn.Module) and 'forward' in obj.__dict__:
        return True
    if hasattr(obj, 'torchdynamo_force_dynamic'):
        return obj.torchdynamo_force_dynamic
    if is_lazy_module(obj):
        return False
    dyn = GenerationTracker.dynamic_classes.get(type(obj)) or GenerationTracker.check(obj)
    return dyn