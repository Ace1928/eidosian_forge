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
def atoms_to_spglib_cell(atoms):
    """Convert atoms into data suitable for calling spglib."""
    return (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())