from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
def most_faithful_signature(self) -> CppSignature:
    if self.faithful_signature:
        return self.faithful_signature
    else:
        return self.signature