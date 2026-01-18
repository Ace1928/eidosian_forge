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
def search_current_git_hash(arg, world=None):
    """Search for .git directory and current git commit hash.

    Parameters:

    arg: str (directory path) or python module
        .git directory is searched from the parent directory of
        the given directory or module.
    """
    if world is None:
        from ase.parallel import world
    if world.rank != 0:
        return None
    if isinstance(arg, str):
        dpath = arg
    else:
        dpath = os.path.dirname(arg.__file__)
    dpath = os.path.realpath(dpath)
    dpath = os.path.dirname(dpath)
    git_dpath = os.path.join(dpath, '.git')
    if not os.path.isdir(git_dpath):
        return None
    HEAD_file = os.path.join(git_dpath, 'HEAD')
    if not os.path.isfile(HEAD_file):
        return None
    with open(HEAD_file, 'r') as fd:
        line = fd.readline().strip()
    if line.startswith('ref: '):
        ref = line[5:]
        ref_file = os.path.join(git_dpath, ref)
    else:
        ref_file = HEAD_file
    if not os.path.isfile(ref_file):
        return None
    with open(ref_file, 'r') as fd:
        line = fd.readline().strip()
    if all((c in string.hexdigits for c in line)):
        return line
    return None