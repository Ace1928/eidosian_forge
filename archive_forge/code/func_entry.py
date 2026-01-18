import os
from pathlib import Path
import stat
from itertools import islice, chain
from typing import Iterable, Optional, List, TextIO
from .translations import _
from .filelock import FileLock
@property
def entry(self) -> str:
    """The current entry, which may be the saved line"""
    return self.entries[-self.index] if self.index else self.saved_line