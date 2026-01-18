from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch
from sympy.testing.pytest import XFAIL
def make_dirs(path):
    """ Create directories (equivalent of ``mkdir -p``). """
    if path[-1] == '/':
        parent = os.path.dirname(path[:-1])
    else:
        parent = os.path.dirname(path)
    if len(parent) > 0:
        if not os.path.exists(parent):
            make_dirs(parent)
    if not os.path.exists(path):
        os.mkdir(path, 511)
    else:
        assert os.path.isdir(path)