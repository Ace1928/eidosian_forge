from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
def ptr_type(self) -> str:
    args_str = ', '.join((a.defn() for a in self.arguments()))
    return f'{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_str})'