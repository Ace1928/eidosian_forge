import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
@doesnt_support_saved_tensors_hooks
def restore_vmap(func, in_dims, batch_size, randomness):

    def inner(*args, **kwargs):
        vmap_level = _vmap_increment_nesting(batch_size, randomness)
        try:
            batched_inputs = wrap_batched(args, in_dims, vmap_level)
            batched_outputs = func(*batched_inputs, **kwargs)
            return unwrap_batched(batched_outputs, vmap_level)
        finally:
            _vmap_decrement_nesting()
    return inner