from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
from torchgen.api.types import Binding, CType, NamedCType
from torchgen.model import (
def argumenttype_evalue_convert(self, t: Type, arg_name: str, *, mutable: bool=False) -> Tuple[str, CType, List[str], List[str]]:
    """
        Takes in the type, name and mutability corresponding to an argument, and generates a tuple of:
        (1) the C++ code necessary to unbox the argument
        (2) A Binding corresponding to the newly created unboxed variable, including variable name and its CType
        :param t: a `Type` of an argument
        :param arg_name: argument name
        :param mutable: boolean for whether this argument type is mutable
        :return: unboxed result
        """
    ctype = self.argument_type_gen(t, mutable=mutable, binds=arg_name).type
    if isinstance(t, BaseType):
        out_name = f'{arg_name}_base'
        code, decl = self._gen_code_base_type(arg_name=arg_name, out_name=out_name, ctype=ctype)
    elif isinstance(t, OptionalType):
        out_name = f'{arg_name}_opt_out'
        code, decl = self._gen_code_optional_type(arg_name=arg_name, out_name=out_name, t=t, ctype=ctype)
    elif isinstance(t, ListType):
        out_name = f'{arg_name}_list_out'
        code, decl = self._gen_code_list_type(arg_name=arg_name, out_name=out_name, t=t, ctype=ctype)
    else:
        raise Exception(f'Cannot handle type {t}. arg_name: {arg_name}')
    return (out_name, ctype, code, decl)