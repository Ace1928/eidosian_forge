from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch
from sympy.testing.pytest import XFAIL
def sha256_of_string(string):
    """ Computes the SHA256 hash of a string. """
    sh = sha256()
    sh.update(string)
    return sh