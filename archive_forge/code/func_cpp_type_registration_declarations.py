from dataclasses import dataclass
from typing import Dict
from torchgen.api.types import (
from torchgen.model import BaseTy
def cpp_type_registration_declarations(self) -> str:
    return f'torch::executor::ArrayRef<{self.elem.cpp_type_registration_declarations()}>'