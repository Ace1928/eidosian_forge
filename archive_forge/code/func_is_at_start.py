import os
from pathlib import Path
import stat
from itertools import islice, chain
from typing import Iterable, Optional, List, TextIO
from .translations import _
from .filelock import FileLock
@property
def is_at_start(self) -> bool:
    return self.index == 0