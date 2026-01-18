from __future__ import annotations
import json
import os
from enum import Enum, unique
from typing import TYPE_CHECKING
from monty.json import MontyEncoder
@staticmethod
def all_kinds():
    """List of strings with the libxc kinds.
        Also in this case, the string is obtained by remove the XC_ prefix.
        XC_CORRELATION --> CORRELATION.
        """
    return sorted({d['Kind'] for d in _all_xcfuncs.values()})