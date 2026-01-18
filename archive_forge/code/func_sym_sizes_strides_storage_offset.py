import contextlib
import warnings
import weakref
from typing import ContextManager, List, Optional, Tuple, TYPE_CHECKING
import torch
from torch._C._functorch import (
from torch._guards import Source
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from torch.utils.weak import WeakIdRef
import torch._prims_common as utils
def sym_sizes_strides_storage_offset(t, src) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    if shape_env is not None:
        if isinstance(t, FakeTensor) and t.fake_mode.shape_env is shape_env:
            return (t.size(), t.stride(), t.storage_offset())
        else:
            return shape_env.create_symbolic_sizes_strides_storage_offset(t, src, symbolic_context=symbolic_context)
    else:
        assert symbolic_context is None
    return (t.size(), t.stride(), t.storage_offset())