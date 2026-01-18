from __future__ import annotations
import json
import os
from enum import Enum, unique
from typing import TYPE_CHECKING
from monty.json import MontyEncoder
@property
def is_k_kind(self) -> bool:
    """True if this is a kinetic functional."""
    return self.kind == 'KINETIC'