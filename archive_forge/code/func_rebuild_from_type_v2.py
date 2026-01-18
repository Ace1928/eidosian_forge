import os
import pickle
import warnings
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, OrderedDict, Sequence, Set, Tuple, Union
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch._C import _TensorMeta
from torch.nn import Parameter
from typing_extensions import override
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.types import _PATH, _Stateful
@classmethod
def rebuild_from_type_v2(cls, func: Callable, new_type: _TensorMeta, args: tuple, state: dict, *, archiveinfo: Optional['_LazyLoadingUnpickler']=None) -> Any:
    ret = func(*args)
    if isinstance(ret, _NotYetLoadedTensor):
        old_lt = ret._load_tensor

        def _load_tensor() -> Any:
            t = old_lt()
            return torch._tensor._rebuild_from_type_v2(lambda: t, new_type, (), state)
        ret._load_tensor = _load_tensor
        return ret
    return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)