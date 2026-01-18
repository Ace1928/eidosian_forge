from __future__ import annotations
import json
import os
from enum import Enum, unique
from typing import TYPE_CHECKING
from monty.json import MontyEncoder
@staticmethod
def all_families():
    """List of strings with the libxc families.
        Note that XC_FAMILY if removed from the string e.g. XC_FAMILY_LDA becomes LDA.
        """
    return sorted({d['Family'] for d in _all_xcfuncs.values()})