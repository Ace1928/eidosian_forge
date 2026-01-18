import fnmatch
import os
import string
import sys
from typing import List, Sequence, Iterable, Optional
from .errors import InvalidPathError
def strip_parent_path(path: str, parent_path: Optional[str]) -> str:
    """Remove a parent path from a path."""
    stripped_path = path
    if parent_path and path.startswith(parent_path):
        stripped_path = path[len(parent_path):]
    return stripped_path