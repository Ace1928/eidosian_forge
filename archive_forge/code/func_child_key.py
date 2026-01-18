from __future__ import annotations
from .base import *
@property
def child_key(self) -> Optional[str]:
    """
        Returns the child key for the index
        """
    return self.primary_key.split('.', 1)[-1] if '.' in self.primary_key else None