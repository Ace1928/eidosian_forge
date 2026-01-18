from dataclasses import dataclass
from typing import Dict, Optional, Tuple
def merge_debug_info(lhs: Optional[Tuple[str, ...]], rhs: Optional[Tuple[str, ...]]) -> Optional[Tuple[str, ...]]:
    if lhs is None and rhs is None:
        return None
    return tuple(set((lhs or ()) + (rhs or ())))