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
def gen_maybe_create_proxy_helper(backend_index: BackendIndex) -> List[str]:
    _, empty_strided_impl = gen_empty_impl_names(backend_index)
    return [] if empty_strided_impl is None else [f'\nc10::optional<Tensor> maybe_create_proxy(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {{\n  if (out.strides() != strides) {{\n    return {empty_strided_impl}(sizes, strides, options);\n  }}\n  return c10::nullopt;\n}}\n']