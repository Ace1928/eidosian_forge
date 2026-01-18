import functools
import weakref
import torch.nn
from torch.nn import Module
from .utils import ExactWeakKeyDictionary, is_lazy_module
@classmethod
def get_generation_value(cls, obj):
    if obj not in cls.generation_values:
        return -1
    return cls.generation_values[obj]