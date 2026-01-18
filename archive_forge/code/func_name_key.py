from __future__ import annotations
import contextlib
from abc import ABC
from typing import (
@property
def name_key(self) -> str:
    """
        Returns the key for the index
        """
    return f'{self.name_prefix}.{self.kdb_type}.{self.name}'