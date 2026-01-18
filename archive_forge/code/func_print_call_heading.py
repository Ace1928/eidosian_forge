import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def print_call_heading(self, name_size, column_title):
    print('Function '.ljust(name_size) + column_title, file=self.stream)
    subheader = False
    for cc, nc, tt, ct, callers in self.stats.values():
        if callers:
            value = next(iter(callers.values()))
            subheader = isinstance(value, tuple)
            break
    if subheader:
        print(' ' * name_size + '    ncalls  tottime  cumtime', file=self.stream)