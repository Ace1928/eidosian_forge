import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def calc_callees(self):
    if self.all_callees:
        return
    self.all_callees = all_callees = {}
    for func, (cc, nc, tt, ct, callers) in self.stats.items():
        if not func in all_callees:
            all_callees[func] = {}
        for func2, caller in callers.items():
            if not func2 in all_callees:
                all_callees[func2] = {}
            all_callees[func2][func] = caller
    return