import sys
import os
from pathlib import Path
import io
def is_pathlib_path(obj):
    """
    Check whether obj is a `pathlib.Path` object.

    Prefer using ``isinstance(obj, os.PathLike)`` instead of this function.
    """
    return isinstance(obj, Path)