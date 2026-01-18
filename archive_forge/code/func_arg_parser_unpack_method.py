from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def arg_parser_unpack_method(t: Type, default: Optional[str], default_init: Optional[str], *, symint: bool=True) -> str:
    has_default_init = default_init is not None
    if has_default_init and str(t) not in ('ScalarType?', 'ScalarType', 'Device', 'Device?', 'Layout', 'Layout?', 'bool', 'bool?'):
        raise RuntimeError(f"type '{t}' does not supported unpacking with default")
    if isinstance(t, BaseType):
        if t.name in [BaseTy.Tensor, BaseTy.Stream, BaseTy.Storage, BaseTy.Scalar, BaseTy.Dimname]:
            return t.name.name.lower()
        elif t.name == BaseTy.ScalarType:
            return 'scalartypeWithDefault' if has_default_init else 'scalartype'
        elif t.name == BaseTy.Device:
            return 'deviceWithDefault' if has_default_init else 'device'
        elif t.name == BaseTy.DeviceIndex:
            return 'toInt64'
        elif t.name == BaseTy.int:
            return 'toInt64'
        elif t.name == BaseTy.SymInt:
            return 'toSymInt' if symint else 'toInt64'
        elif t.name == BaseTy.bool:
            return 'toBoolWithDefault' if has_default_init else 'toBool'
        elif t.name == BaseTy.float:
            return 'toDouble'
        elif t.name == BaseTy.str:
            return 'stringView'
        elif t.name == BaseTy.Layout:
            return 'layoutWithDefault' if has_default_init else 'layout'
        elif t.name == BaseTy.MemoryFormat:
            return 'memoryformat'
    elif isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            return 'optionalTensor'
        elif str(t.elem) == 'Generator':
            return 'generator'
        elif str(t.elem) == 'Dimname[]':
            return 'toDimnameListOptional'
        elif not has_default_init and default in (None, 'None', 'c10::nullopt'):
            return arg_parser_unpack_method(t.elem, None, None, symint=symint) + 'Optional'
        else:
            return arg_parser_unpack_method(t.elem, default, default_init, symint=symint)
    elif isinstance(t, ListType):
        if str(t.elem) == 'Tensor':
            return f'tensorlist_n<{t.size}>' if t.size is not None else 'tensorlist'
        elif str(t.elem) == 'Tensor?':
            return 'list_of_optional_tensors'
        elif str(t.elem) == 'Dimname':
            return 'dimnamelist'
        elif str(t.elem) == 'int':
            return 'intlist'
        elif str(t.elem) == 'float':
            return 'doublelist'
        elif str(t.elem) == 'SymInt':
            return 'symintlist' if symint else 'intlist'
        elif str(t.elem) == 'Scalar':
            return 'scalarlist'
    raise RuntimeError(f"type '{t}' is not supported by PythonArgParser")