from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@property
def is_top(self) -> bool:
    """
        Return True if panel is at the top
        """
    return self.row == 1