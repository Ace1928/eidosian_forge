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
def del_ten():
    self_ref = self_weak_ref()
    if self_ref is None:
        return
    self_ref.tensor_memo.pop(tensor_ref_key, None)
    if weak_st and weak_st.expired():
        self_ref.storage_memo.pop(weak_st, None)
    elif weak_st is not None:
        self_ref.maybe_storages_to_delete.append(weak_st)