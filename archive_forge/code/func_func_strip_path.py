import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def func_strip_path(func_name):
    filename, line, name = func_name
    return (os.path.basename(filename), line, name)