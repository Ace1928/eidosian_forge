import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def print_title(self):
    print('   ncalls  tottime  percall  cumtime  percall', end=' ', file=self.stream)
    print('filename:lineno(function)', file=self.stream)