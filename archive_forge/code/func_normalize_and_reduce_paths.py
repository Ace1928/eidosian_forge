import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def normalize_and_reduce_paths(paths):
    """Return a list of normalized paths with duplicates removed.

    The current order of paths is maintained.
    """
    reduced_paths = []
    for p in paths:
        np = os.path.normpath(p)
        if np not in reduced_paths:
            reduced_paths.append(np)
    return reduced_paths