import functools
import weakref
import torch.nn
from torch.nn import Module
from .utils import ExactWeakKeyDictionary, is_lazy_module
@functools.wraps(original_setattr)
def custom_setattr(self, key, value):
    try:
        MutationTracker.db[self].on_mutation(key)
    except KeyError:
        pass
    return original_setattr(self, key, value)