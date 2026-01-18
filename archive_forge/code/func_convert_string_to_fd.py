import errno
import functools
import os
import io
import pickle
import sys
import time
import string
import warnings
from importlib import import_module
from math import sin, cos, radians, atan2, degrees
from contextlib import contextmanager, ExitStack
from math import gcd
from pathlib import PurePath, Path
import re
import numpy as np
from ase.formula import formula_hill, formula_metal
@deprecated('convert_string_to_fd does not facilitate proper resource management.  Please use e.g. ase.utils.IOContext class instead.')
def convert_string_to_fd(name, world=None):
    """Create a file-descriptor for text output.

    Will open a file for writing with given name.  Use None for no output and
    '-' for sys.stdout.
    """
    if world is None:
        from ase.parallel import world
    if name is None or world.rank != 0:
        return open(os.devnull, 'w')
    if name == '-':
        return sys.stdout
    if isinstance(name, (str, PurePath)):
        return open(str(name), 'w')
    return name