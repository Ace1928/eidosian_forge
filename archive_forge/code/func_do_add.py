import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def do_add(self, line):
    if self.stats:
        try:
            self.stats.add(line)
        except OSError as e:
            print('Failed to load statistics for %s: %s' % (line, e), file=self.stream)
    else:
        print('No statistics object is loaded.', file=self.stream)
    return 0