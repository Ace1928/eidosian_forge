import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import cast, NoReturn, Optional
import torch._guards
from . import config
from .config import is_fbcode
from .utils import counters
import logging
def remove_from_stats(self):
    assert self.category is not None
    counters[self.category][self.msg] -= 1
    if counters[self.category][self.msg] <= 0:
        del counters[self.category][self.msg]