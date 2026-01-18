from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
def dispatcher_exprs(self) -> List[Expr]:
    return translate.translate(self.arguments(), dispatcher.arguments(self.func), method=False)