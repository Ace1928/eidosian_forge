from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch
from sympy.testing.pytest import XFAIL
def pyx_is_cplus(path):
    """
    Inspect a Cython source file (.pyx) and look for comment line like:

    # distutils: language = c++

    Returns True if such a file is present in the file, else False.
    """
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') and '=' in line:
                splitted = line.split('=')
                if len(splitted) != 2:
                    continue
                lhs, rhs = splitted
                if lhs.strip().split()[-1].lower() == 'language' and rhs.strip().split()[0].lower() == 'c++':
                    return True
    return False