from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch
from sympy.testing.pytest import XFAIL
def find_binary_of_command(candidates):
    """ Finds binary first matching name among candidates.

    Calls ``which`` from shutils for provided candidates and returns
    first hit.

    Parameters
    ==========

    candidates : iterable of str
        Names of candidate commands

    Raises
    ======

    CompilerNotFoundError if no candidates match.
    """
    from shutil import which
    for c in candidates:
        binary_path = which(c)
        if c and binary_path:
            return (c, binary_path)
    raise CompilerNotFoundError('No binary located for candidates: {}'.format(candidates))