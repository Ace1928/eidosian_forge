from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@property
def is_bottom(self) -> bool:
    """
        Return True if Panel is at the bottom
        """
    return self.row == self.nrow