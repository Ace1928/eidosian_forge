import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def line_with_cursor(cursor_offset: int, line: str) -> str:
    return line[:cursor_offset] + '|' + line[cursor_offset:]