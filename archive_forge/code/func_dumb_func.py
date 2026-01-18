import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def dumb_func(cursor_offset, line):
    return LinePart(0, 2, 'ab')