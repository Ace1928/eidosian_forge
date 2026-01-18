from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@property
def is_left(self) -> bool:
    """
        Return True if panel is on the left
        """
    return self.col == 1