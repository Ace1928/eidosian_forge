import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
def gen_empty_impl_names(backend_index: BackendIndex) -> Tuple[Optional[str], Optional[str]]:
    empty_impl = None
    empty_strided_impl = None
    if backend_index.dispatch_key in (DispatchKey.Meta, DispatchKey.CPU, DispatchKey.CUDA, DispatchKey.MPS):
        dispatch = str(backend_index.dispatch_key).lower()
        empty_impl = f'at::detail::empty_{dispatch}'
        empty_strided_impl = f'at::detail::empty_strided_{dispatch}'
    elif backend_index.dispatch_key in (DispatchKey.CompositeExplicitAutogradNonFunctional, DispatchKey.QuantizedCPU, DispatchKey.QuantizedCUDA):
        empty_impl = 'at::empty'
        empty_strided_impl = 'at::empty_strided'
    return (empty_impl, empty_strided_impl)