from __future__ import annotations
import json
import os
from enum import Enum, unique
from typing import TYPE_CHECKING
from monty.json import MontyEncoder
@property
def info_dict(self):
    """Dictionary with metadata. see libxc_docs.json."""
    return _all_xcfuncs[self.value]