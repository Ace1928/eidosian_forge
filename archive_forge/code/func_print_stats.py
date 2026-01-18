import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def print_stats(self, *amount):
    for filename in self.files:
        print(filename, file=self.stream)
    if self.files:
        print(file=self.stream)
    indent = ' ' * 8
    for func in self.top_level:
        print(indent, func_get_function_name(func), file=self.stream)
    print(indent, self.total_calls, 'function calls', end=' ', file=self.stream)
    if self.total_calls != self.prim_calls:
        print('(%d primitive calls)' % self.prim_calls, end=' ', file=self.stream)
    print('in %.3f seconds' % self.total_tt, file=self.stream)
    print(file=self.stream)
    width, list = self.get_print_list(amount)
    if list:
        self.print_title()
        for func in list:
            self.print_line(func)
        print(file=self.stream)
        print(file=self.stream)
    return self