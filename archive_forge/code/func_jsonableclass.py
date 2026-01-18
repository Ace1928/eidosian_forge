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
def jsonableclass(cls):
    cls.ase_objtype = name
    if not hasattr(cls, 'todict'):
        raise TypeError('Class must implement todict()')
    cls.write = write_json
    cls.read = read_json
    return cls