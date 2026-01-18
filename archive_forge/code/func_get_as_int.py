from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from torch.distributed import Store
def get_as_int(self, key: str, default: Optional[int]=None) -> Optional[int]:
    """Return the value for ``key`` as an ``int``."""
    value = self.get(key, default)
    if value is None:
        return value
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"The rendezvous configuration option '{key}' does not represent a valid integer value.") from e