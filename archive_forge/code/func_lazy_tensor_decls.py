import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
def lazy_tensor_decls(self, func: NativeFunction, schema: LazyIrSchema) -> str:
    value_args = schema.filtered_args(values=True, scalars=False)
    lazy_tensor_decls: List[str] = []
    for arg in value_args:
        if arg.is_wrapped_scalar:
            if isinstance(arg.lazy_type, OptionalCType):
                lazy_tensor_decls.append(f'auto node_{arg.name} = {arg.name} ?\n                c10::make_optional(torch::lazy::LazyGraphExecutor::Get()->\n                    GetIrValueForScalarFromCodegen(*{arg.name}, *common_device)):\n                c10::nullopt;')
            else:
                lazy_tensor_decls.append(f'auto node_{arg.name} = torch::lazy::LazyGraphExecutor::Get()->\n                            GetIrValueForScalarFromCodegen({arg.name}, *common_device);')
        elif arg.is_symint_or_list:
            continue
        elif isinstance(arg.lazy_type, BaseCType):
            if arg.lazy_type.type is tensorListValueT:
                lazy_tensor_decls.append(f'auto lazy_{arg.name}_tensorlist = {self.backend_namespace}::{self.get_tensorlist}({arg.name});')
            else:
                lazy_tensor_decls.append(f'{self.lazy_tensor_ptr} lazy_{arg.name} = {self.backend_namespace}::{self.get_tensor_or_wrap_number}({arg.name}, *common_device);')
        elif isinstance(arg.lazy_type, OptionalCType):
            assert arg.lazy_type.elem == BaseCType(getValueT()), arg.lazy_type.elem
            lazy_tensor_decls.append(f'{self.lazy_tensor_ptr} lazy_{arg.name} = {self.backend_namespace}::{self.try_get_tensor}({arg.name}.value_or(at::Tensor()));')
        else:
            raise AssertionError(f'TODO not sure if there are other valid types to handle here ({arg.lazy_type})')
    return '\n        '.join(lazy_tensor_decls)