from dataclasses import dataclass
from typing import Dict
from torchgen.api.types import (
from torchgen.model import BaseTy
def remove_const_ref(self) -> 'CType':
    return ArrayRefCType(self.elem.remove_const_ref())