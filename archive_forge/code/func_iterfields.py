from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
def iterfields(self) -> Iterator[tuple[str, Optional[str]]]:
    """
        Return an iterator of all (field, value) pairs
        """
    return ((f.name, getattr(self, f.name)) for f in fields(self))