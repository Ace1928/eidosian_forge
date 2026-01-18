from typing import List, Tuple
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignatureGroup, CType
from torchgen.model import (
def argumenttype_ivalue_convert(t: Type, arg_name: str, *, mutable: bool=False) -> Tuple[str, CType, List[str], List[str]]:
    ctype = cpp.argumenttype_type(t=t, mutable=mutable, binds=arg_name, symint=False).type
    if isinstance(t, BaseType):
        out_name = f'{arg_name}_base'
        code, decl = _gen_code_base_type(arg_name=arg_name, out_name=out_name, ctype=ctype)
    elif isinstance(t, OptionalType):
        out_name = f'{arg_name}_opt_out'
        code, decl = _gen_code_optional_type(arg_name=arg_name, out_name=out_name, t=t, ctype=ctype)
    elif isinstance(t, ListType):
        out_name = f'{arg_name}_list_out'
        code, decl = _gen_code_list_type(arg_name=arg_name, out_name=out_name, t=t, ctype=ctype)
    else:
        raise Exception(f'Cannot handle type {t}. arg_name: {arg_name}')
    return (out_name, ctype, code, decl)