import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def sort_stats(self, *field):
    if not field:
        self.fcn_list = 0
        return self
    if len(field) == 1 and isinstance(field[0], int):
        field = [{-1: 'stdname', 0: 'calls', 1: 'time', 2: 'cumulative'}[field[0]]]
    elif len(field) >= 2:
        for arg in field[1:]:
            if type(arg) != type(field[0]):
                raise TypeError("Can't have mixed argument type")
    sort_arg_defs = self.get_sort_arg_defs()
    sort_tuple = ()
    self.sort_type = ''
    connector = ''
    for word in field:
        if isinstance(word, SortKey):
            word = word.value
        sort_tuple = sort_tuple + sort_arg_defs[word][0]
        self.sort_type += connector + sort_arg_defs[word][1]
        connector = ', '
    stats_list = []
    for func, (cc, nc, tt, ct, callers) in self.stats.items():
        stats_list.append((cc, nc, tt, ct) + func + (func_std_string(func), func))
    stats_list.sort(key=cmp_to_key(TupleComp(sort_tuple).compare))
    self.fcn_list = fcn_list = []
    for tuple in stats_list:
        fcn_list.append(tuple[-1])
    return self