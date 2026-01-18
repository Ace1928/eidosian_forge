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
def get_tensor_memo(self, t):
    return self.tensor_memo.get(WeakIdRef(t), None)