import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def get_top_level_stats(self):
    for func, (cc, nc, tt, ct, callers) in self.stats.items():
        self.total_calls += nc
        self.prim_calls += cc
        self.total_tt += tt
        if ('jprofile', 0, 'profiler') in callers:
            self.top_level.add(func)
        if len(func_std_string(func)) > self.max_name_len:
            self.max_name_len = len(func_std_string(func))