from dataclasses import dataclass
from typing import List, Optional, Set
import torchgen.api.cpp as aten_cpp
from torchgen.api.types import Binding, CType
from torchgen.model import FunctionSchema, NativeFunction
from .types import contextArg
from torchgen.executorch.api import et_cpp
def decl(self, name: Optional[str]=None, *, include_context: bool=True) -> str:
    args_str = ', '.join((a.decl() for a in self.arguments(include_context=include_context)))
    if name is None:
        name = self.name()
    return f'{self.returns_type().cpp_type()} {name}({args_str})'