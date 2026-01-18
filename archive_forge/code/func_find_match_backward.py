import os
from pathlib import Path
import stat
from itertools import islice, chain
from typing import Iterable, Optional, List, TextIO
from .translations import _
from .filelock import FileLock
def find_match_backward(self, search_term: str, include_current: bool=False) -> int:
    add = 0 if include_current else 1
    start = self.index + add
    for idx, val in enumerate(islice(self.entries_by_index, start, None)):
        if val.startswith(search_term):
            return idx + add
    return 0