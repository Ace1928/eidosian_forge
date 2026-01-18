from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
from torchgen.api.types import Binding, CType, NamedCType
from torchgen.model import (
def convert_arguments(self, args: Sequence[Binding]) -> Tuple[List[Binding], List[str]]:
    code_list = [f'EValue& {args[i].name} = *stack[{i}];' for i in range(len(args))]
    binding_list = []
    for arg in args:
        if not isinstance(arg.argument, Argument):
            raise Exception(f'Unexpected argument type, expecting `Argument` but got {arg}')
        argument: Argument = arg.argument
        unboxed_name, _, code, decl = self.argumenttype_evalue_convert(argument.type, argument.name, mutable=argument.is_write)
        code_list.extend(decl)
        code_list.extend(code)
        binding_list.append(arg.with_name(unboxed_name))
    return (binding_list, code_list)