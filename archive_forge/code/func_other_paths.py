from __future__ import annotations
import contextlib
import logging
import math
import os
import pathlib
import re
import sys
import tempfile
from functools import partial
from hashlib import md5
from importlib.metadata import version
from typing import (
from urllib.parse import urlsplit
def other_paths(paths: list[str], path2: str | list[str], exists: bool=False, flatten: bool=False) -> list[str]:
    """In bulk file operations, construct a new file tree from a list of files

    Parameters
    ----------
    paths: list of str
        The input file tree
    path2: str or list of str
        Root to construct the new list in. If this is already a list of str, we just
        assert it has the right number of elements.
    exists: bool (optional)
        For a str destination, it is already exists (and is a dir), files should
        end up inside.
    flatten: bool (optional)
        Whether to flatten the input directory tree structure so that the output files
        are in the same directory.

    Returns
    -------
    list of str
    """
    if isinstance(path2, str):
        path2 = path2.rstrip('/')
        if flatten:
            path2 = ['/'.join((path2, p.split('/')[-1])) for p in paths]
        else:
            cp = common_prefix(paths)
            if exists:
                cp = cp.rsplit('/', 1)[0]
            if not cp and all((not s.startswith('/') for s in paths)):
                path2 = ['/'.join([path2, p]) for p in paths]
            else:
                path2 = [p.replace(cp, path2, 1) for p in paths]
    else:
        assert len(paths) == len(path2)
    return path2