from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module
def bundle_large_tensor(t):
    """Wrap a tensor to allow bundling regardless of size."""
    return InflatableArg(value=t, fmt='{}')