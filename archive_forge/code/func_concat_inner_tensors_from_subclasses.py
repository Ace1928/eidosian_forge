from typing import Any, List, Optional, Tuple, Union
import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from .schemas import SubclassCreationMeta, ViewAndMutationMeta
from .utils import strict_zip
def concat_inner_tensors_from_subclasses(xs):
    xs_inner = []
    for x in xs:
        if isinstance(x, Tensor) and is_traceable_wrapper_subclass(x):
            attrs, _ = x.__tensor_flatten__()
            xs_inner += [getattr(x, attr) for attr in attrs]
        else:
            xs_inner += [x]
    return xs_inner