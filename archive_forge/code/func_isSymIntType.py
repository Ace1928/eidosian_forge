from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
def isSymIntType(typ: Type) -> bool:
    return isinstance(typ, BaseType) and typ.name == BaseTy.SymInt