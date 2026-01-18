import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def copy_with_new_str(self, new_str: str) -> 'FmtStr':
    """Copies the current FmtStr's attributes while changing its string."""
    old_atts = {att: value for bfs in self.chunks for att, value in bfs.atts.items()}
    return FmtStr(Chunk(new_str, old_atts))