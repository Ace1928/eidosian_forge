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
def gen_class_set_output_body(self, k: SchemaKind, maybe_create_proxy: bool) -> str:
    if self.backend_index.dispatch_key in [DispatchKey.CUDA, DispatchKey.MPS, DispatchKey.CompositeExplicitAutogradNonFunctional]:
        maybe_set_guard = '\nauto current_device = guard_.current_device();\nif (C10_UNLIKELY(current_device.has_value())) {\n  TORCH_INTERNAL_ASSERT(*current_device == options.device(),\n    "structured kernels don\'t support multi-device outputs");\n} else {\n  guard_.reset_device(options.device());\n}\n'
        maybe_set_guard_line = maybe_set_guard + '\n'
    else:
        maybe_set_guard_line = maybe_set_guard = ''
    if maybe_create_proxy:
        create_proxy = '\nauto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);\nif (C10_UNLIKELY(maybe_proxy.has_value())) {\n    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();\n}\n'
    else:
        create_proxy = ''
    if k is SchemaKind.functional:
        assert self.backend_index.dispatch_key in (DispatchKey.Meta, DispatchKey.CPU, DispatchKey.CUDA, DispatchKey.MPS, DispatchKey.CompositeExplicitAutogradNonFunctional)
        return f'{maybe_set_guard_line}\noutputs_[output_idx] = create_out(sizes, strides, options);'
    elif k is SchemaKind.inplace:
        return f'{maybe_set_guard_line}\nconst auto& out = outputs_[output_idx].get();\ncheck_inplace(out, sizes, options);\n{create_proxy}'
    elif k is SchemaKind.out:
        return f'{maybe_set_guard_line}\nconst auto& out = outputs_[output_idx].get();\nresize_out(out, sizes, strides, options);\n{create_proxy}'
    elif k is SchemaKind.mutable or k is SchemaKind.scratch:
        raise AssertionError(f'{k} structured operators are currently not supported')
    else:
        assert_never(k)