from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Type
import torch
import torch.fx
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
from .virtualized import V
def is_fake_tensor_same(new, old):
    if type(new) != type(old):
        return False
    if isinstance(new, (list, tuple)):
        if len(new) != len(old):
            return False
        return all((is_fake_tensor_same(new_i, old_i) for new_i, old_i in zip(new, old)))
    assert isinstance(new, torch.Tensor)
    if new.shape != old.shape or new.layout != old.layout:
        return False
    if new.layout == torch.strided and new.stride() != old.stride():
        return False
    if get_storage(new) == get_storage(old):
        return True
    if existing_storages[get_storage(old)] == 1 and get_storage(new) not in existing_storages:
        return True
    return False