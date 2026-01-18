import fnmatch
import os
import string
import sys
from typing import List, Sequence, Iterable, Optional
from .errors import InvalidPathError
def is_valid_filename(filename: str) -> bool:
    """Verifies if a filename does not contain illegal character sequences"""
    valid = True
    valid = valid and os.pardir not in filename
    valid = valid and os.sep not in filename
    if os.altsep:
        valid = valid and os.altsep not in filename
    return valid