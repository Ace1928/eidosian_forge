from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def kernel_name(g: NativeFunctionsGroup, dispatch_key: DispatchKey) -> str:
    return schema_kernel_name(g.out.func, dispatch_key)