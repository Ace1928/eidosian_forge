import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def complete_sort(self, text, *args):
    return [a for a in Stats.sort_arg_dict_default if a.startswith(text)]