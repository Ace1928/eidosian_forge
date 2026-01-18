from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def opmath_type(scalar_t: BaseCppType) -> BaseCppType:
    if scalar_t == api_types.scalar_t:
        return api_types.opmath_t
    raise NotImplementedError