from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def argument_type_size(t: Type) -> Optional[int]:
    l = t.is_list_like()
    if l is not None and str(l.elem) != 'bool':
        return l.size
    else:
        return None