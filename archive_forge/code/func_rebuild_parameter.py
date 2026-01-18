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
def rebuild_parameter(cls, data: Any, requires_grad: bool, backward_hooks: OrderedDict, *, archiveinfo: Optional['_LazyLoadingUnpickler']=None) -> Union[Tensor, '_NotYetLoadedTensor']:
    if isinstance(data, _NotYetLoadedTensor):
        old_lt = data._load_tensor

        def _load_tensor() -> Parameter:
            t = old_lt()
            return torch._utils._rebuild_parameter(t, requires_grad, backward_hooks)
        data._load_tensor = _load_tensor
        return data
    return torch._utils._rebuild_parameter(data, requires_grad, backward_hooks)