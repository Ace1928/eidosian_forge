import os
from pathlib import Path
import stat
from itertools import islice, chain
from typing import Iterable, Optional, List, TextIO
from .translations import _
from .filelock import FileLock
def save_to(self, fd: TextIO, entries: Optional[List[str]]=None, lines: int=0) -> None:
    if entries is None:
        entries = self.entries
    for line in entries[-lines:]:
        fd.write(line)
        fd.write('\n')