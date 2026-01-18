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
def node_ctor_arg_rvalue_string(arg: LazyArgument) -> str:
    """
    Given a LazyArgument,
    generate a c++ string for materializing an rvalue of that arg for passing into
    a lazy Node constructor.
    """
    if isValueType(arg.lazy_type):
        if isinstance(arg.lazy_type, BaseCType):
            if arg.is_wrapped_scalar:
                return f'node_{arg.name}'
            elif arg.lazy_type.type is tensorListValueT:
                return f'lazy_{arg.name}_tensorlist'
            elif arg.is_symint_or_list:
                return f'GetSymIntValue({arg.name})'
            return f'lazy_{arg.name}->GetIrValue()'
        elif isinstance(arg.lazy_type, OptionalCType):
            if arg.is_symint_or_list:
                return f'{arg.name} ? c10::make_optional(GetSymIntValue(*{arg.name})) : c10::nullopt'
            elif arg.is_wrapped_scalar:
                return f'node_{arg.name}'
            return f'lazy_{arg.name} ? c10::make_optional(lazy_{arg.name}->GetIrValue()) : c10::nullopt'
        else:
            raise AssertionError(f'TODO not sure if there are other valid types to handle here ({arg.lazy_type})')
    elif isinstance(arg.orig_type, ListType) and arg.orig_type.elem == BaseType(BaseTy.SymInt):
        if arg.symint:
            return f'GetSymIntArrayRefValue({arg.name})'
        else:
            return f'std::vector<int64_t>({arg.name}.begin(), {arg.name}.end())'
    elif isinstance(arg.lazy_type, VectorCType) and isinstance(arg.lazy_type.elem, BaseCType):
        return f'std::vector<{arg.lazy_type.elem.type}>({arg.name}.begin(), {arg.name}.end())'
    elif isinstance(arg.lazy_type, OptionalCType) and isinstance(arg.lazy_type.elem, VectorCType) and isinstance(arg.lazy_type.elem.elem, BaseCType):
        return f'torch::lazy::ToOptionalVector<{arg.lazy_type.elem.elem.type}>({arg.name})'
    else:
        return f'{arg.name}'