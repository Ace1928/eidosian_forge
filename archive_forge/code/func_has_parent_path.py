import fnmatch
import os
import string
import sys
from typing import List, Sequence, Iterable, Optional
from .errors import InvalidPathError
def has_parent_path(path: str, parent_path: Optional[str]) -> bool:
    """Verifies if path falls under parent_path."""
    check = True
    if parent_path:
        check = os.path.commonpath([path, parent_path]) == parent_path
    return check