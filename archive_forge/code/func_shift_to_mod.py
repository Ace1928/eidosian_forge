import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def shift_to_mod(self, pos):
    if pos < self.orig_pos - 1:
        return 0
    elif pos > self.orig_pos + self.orig_range:
        return self.mod_range - self.orig_range
    else:
        return self.shift_to_mod_lines(pos)