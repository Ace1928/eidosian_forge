import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def strip_dirs(self):
    oldstats = self.stats
    self.stats = newstats = {}
    max_name_len = 0
    for func, (cc, nc, tt, ct, callers) in oldstats.items():
        newfunc = func_strip_path(func)
        if len(func_std_string(newfunc)) > max_name_len:
            max_name_len = len(func_std_string(newfunc))
        newcallers = {}
        for func2, caller in callers.items():
            newcallers[func_strip_path(func2)] = caller
        if newfunc in newstats:
            newstats[newfunc] = add_func_stats(newstats[newfunc], (cc, nc, tt, ct, newcallers))
        else:
            newstats[newfunc] = (cc, nc, tt, ct, newcallers)
    old_top = self.top_level
    self.top_level = new_top = set()
    for func in old_top:
        new_top.add(func_strip_path(func))
    self.max_name_len = max_name_len
    self.fcn_list = None
    self.all_callees = None
    return self