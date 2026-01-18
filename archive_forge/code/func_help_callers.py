import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def help_callers(self):
    print('Print callers statistics from the current stat object.', file=self.stream)
    self.generic_help()