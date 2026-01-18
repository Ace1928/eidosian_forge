import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def pos_in_mod(self, position):
    newpos = position
    for hunk in self.hunks:
        shift = hunk.shift_to_mod(position)
        if shift is None:
            return None
        newpos += shift
    return newpos