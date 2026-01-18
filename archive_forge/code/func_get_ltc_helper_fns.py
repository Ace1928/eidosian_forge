import argparse
import os
import pathlib
import re
from collections import Counter, namedtuple
from typing import (
import yaml
import torchgen.dest as dest
from torchgen.api.lazy import setValueT
from torchgen.api.types import BaseCppType
from torchgen.dest.lazy_ir import GenLazyIR, GenLazyNativeFuncDefinition, GenTSLazyIR
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import NativeFunction, NativeFunctionsGroup, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, FileManager, NamespaceHelper
from torchgen.yaml_utils import YamlLoader
from .gen_backend_stubs import (
def get_ltc_helper_fns() -> str:
    return "at::Tensor to_meta(const at::Tensor& tensor) {\n  // undefined tensors can't be converted to the meta device, since they don't have sizes/strides\n  if (!tensor.defined()) return tensor;\n  auto out = at::native::empty_strided_meta_symint(tensor.sym_sizes(), tensor.sym_strides(), /*dtype=*/c10::make_optional(tensor.scalar_type()), /*layout=*/c10::make_optional(tensor.layout()), /*device=*/c10::make_optional(c10::Device(c10::kMeta)), /*pin_memory=*/c10::nullopt);\n  // needs to handle wrapped numbers, so dtype promotion works properly.\n  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {\n    out.unsafeGetTensorImpl()->set_wrapped_number(true);\n  }\n  return out;\n}\nc10::optional<at::Tensor> to_meta(const c10::optional<at::Tensor>& tensor) {\n  if (tensor.has_value()) {\n    return to_meta(*tensor);\n  }\n  return c10::nullopt;\n}\n\nstd::vector<at::Tensor> to_meta(at::ITensorListRef t_list) {\n  std::vector<at::Tensor> outs;\n  outs.reserve(t_list.size());\n  for (const auto& tensor : t_list) {\n    outs.push_back(to_meta(tensor));\n  }\n  return outs;\n}\n"