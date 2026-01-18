from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
def setValueT(val: BaseCppType) -> None:
    global _valueT
    _valueT = val