from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
def get_as_bool(self, key: str, default: Optional[bool]=None) -> Optional[bool]:
    """Return the value for ``key`` as a ``bool``."""
    value = self.get(key, default)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    elif isinstance(value, str):
        if value.lower() in ['1', 'true', 't', 'yes', 'y']:
            return True
        if value.lower() in ['0', 'false', 'f', 'no', 'n']:
            return False
    raise ValueError(f"The rendezvous configuration option '{key}' does not represent a valid boolean value.")