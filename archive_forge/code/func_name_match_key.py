from __future__ import annotations
import contextlib
from abc import ABC
from typing import (
@property
def name_match_key(self) -> str:
    """
        Returns the match key for the index - this is used to match all keys for the index
        """
    return f'{self.name_prefix}.{self.kdb_type}.{self.name}:*' if self.name_prefix_enabled else '*'