from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
def getValueT() -> BaseCppType:
    global _valueT
    if not _valueT:
        raise NotImplementedError('The value type needs to be set with setValueT() in run_gen_lazy_tensor()')
    return _valueT