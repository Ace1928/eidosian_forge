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
def gen_resize_out_helper(backend_index: BackendIndex) -> List[str]:
    if backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
        return []
    return ['\nvoid resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {\n  TORCH_CHECK(options.dtype() == out.dtype(),\n      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");\n  TORCH_CHECK(options.device() == out.device(),\n      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");\n  const bool resized = at::native::resize_output(out, sizes);\n  // Only restride if a resize occurred; otherwise we ignore the (advisory)\n  // strides from the meta function and directly use the output tensor\'s\n  // preexisting strides\n  if (resized) {\n    if (!strides.empty()) {\n      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());\n      // TODO: avoid the redispatch here\n      out.as_strided_(sizes, strides);\n    } else if (options.memory_format_opt().has_value()) {\n      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());\n    }\n  }\n}\n']