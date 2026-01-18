import os
import numbers
from pathlib import Path
from typing import Union, Set
import numpy as np
from ase.io.jsonio import encode, decode
from ase.utils import plural
def file_has_fileno(fd):
    """Tell whether file implements fileio() or not.

    array.tofile(fd) works only on files with fileno().
    numpy may write faster to physical files using fileno().

    For files without fileno() we use instead fd.write(array.tobytes()).
    Either way we need to distinguish."""
    try:
        fno = fd.fileno
        fno()
    except (AttributeError, IOError):
        return False
    return True