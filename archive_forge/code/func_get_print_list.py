import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def get_print_list(self, sel_list):
    width = self.max_name_len
    if self.fcn_list:
        stat_list = self.fcn_list[:]
        msg = '   Ordered by: ' + self.sort_type + '\n'
    else:
        stat_list = list(self.stats.keys())
        msg = '   Random listing order was used\n'
    for selection in sel_list:
        stat_list, msg = self.eval_print_amount(selection, stat_list, msg)
    count = len(stat_list)
    if not stat_list:
        return (0, stat_list)
    print(msg, file=self.stream)
    if count < len(self.stats):
        width = 0
        for func in stat_list:
            if len(func_std_string(func)) > width:
                width = len(func_std_string(func))
    return (width + 2, stat_list)