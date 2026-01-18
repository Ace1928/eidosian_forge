from dataclasses import dataclass
from typing import List, Optional, Set
import torchgen.api.cpp as aten_cpp
from torchgen.api.types import Binding, CType
from torchgen.model import FunctionSchema, NativeFunction
from .types import contextArg
from torchgen.executorch.api import et_cpp
def returns_type(self) -> CType:
    return et_cpp.returns_type(self.func.returns)