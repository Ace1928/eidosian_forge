import logging
from typing import Dict, Optional, Tuple, Type
import sympy
from torch.utils._sympy.functions import FloorDiv
def mirror_rel_op(type: Type) -> Optional[Type[sympy.Rel]]:
    return _MIRROR_REL_OP.get(type, None)