import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def iter_inserted(self):
    """Iteraties through inserted lines

        :return: Pair of line number, line
        :rtype: iterator of (int, InsertLine)
        """
    for hunk in self.hunks:
        pos = hunk.mod_pos - 1
        for line in hunk.lines:
            if isinstance(line, InsertLine):
                yield (pos, line)
                pos += 1
            if isinstance(line, ContextLine):
                pos += 1