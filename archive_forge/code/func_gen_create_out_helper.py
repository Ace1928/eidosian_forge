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
def gen_create_out_helper(backend_index: BackendIndex) -> List[str]:
    if backend_index.dispatch_key == DispatchKey.Meta:
        empty_options = 'options.device(at::kMeta)'
    else:
        empty_options = 'options'
    empty_impl, empty_strided_impl = gen_empty_impl_names(backend_index)
    if empty_impl is None:
        return []
    return [f'\nTensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{\n  if (strides.empty()) {{\n      return {empty_impl}(sizes, {empty_options});\n  }} else {{\n      return {empty_strided_impl}(sizes, strides, {empty_options});\n  }}\n}}\n']