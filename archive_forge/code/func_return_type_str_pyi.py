from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def return_type_str_pyi(t: Type) -> str:
    if isinstance(t, OptionalType):
        inner = return_type_str_pyi(t.elem)
        return f'Optional[{inner}]'
    if isinstance(t, BaseType):
        if t.name == BaseTy.Device:
            return '_device'
        elif t.name == BaseTy.Dimname:
            ret = 'Optional[str]'
        else:
            return argument_type_str_pyi(t)
    if isinstance(t, ListType):
        inner = return_type_str_pyi(t.elem)
        return f'List[{inner}]'
    return argument_type_str_pyi(t)