import contextlib
import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, Union
import torchgen.local as local
from torchgen.model import (
from torchgen.utils import context, S, T
@contextlib.contextmanager
def native_function_manager(g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup, NativeFunction]) -> Iterator[None]:
    if isinstance(g, NativeFunctionsGroup):
        f = g.out
    elif isinstance(g, NativeFunctionsViewGroup):
        f = g.view
    else:
        f = g
    with context(lambda: f'in native_functions.yaml line {f.loc}:\n  {f.func}'):
        with local.parametrize(use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors, use_ilistref_for_tensor_lists=f.part_of_structured_group):
            yield