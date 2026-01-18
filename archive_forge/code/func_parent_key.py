from __future__ import annotations
from .base import *
@property
def parent_key(self) -> str:
    """
        Returns the parent key for the index
        """
    return self.primary_key.split('.', 1)[0] if '.' in self.primary_key else self.primary_key