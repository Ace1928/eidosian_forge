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
@functools.wraps(func)
def iofunc(file, *args, **kwargs):
    openandclose = isinstance(file, (str, PurePath))
    fd = None
    try:
        if openandclose:
            fd = open(str(file), self.mode)
        else:
            fd = file
        obj = func(fd, *args, **kwargs)
        return obj
    finally:
        if openandclose and fd is not None:
            fd.close()