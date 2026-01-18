import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def group_parents(self, i):
    """Return list of parent group paths.

                Args:
                    i: (int) return parents of this path.
                Returns:
                    List of the group parents."""
    if i >= self.path_count:
        raise IndexError('bad path index')
    while i < 0:
        i += self.path_count
    lvl = self.paths[i].level
    groups = list(reversed([p for p in self.paths[:i] if p.type == 'group' and p.level < lvl]))
    if groups == []:
        return []
    ngroups = [groups[0]]
    for p in groups[1:]:
        if p.level >= ngroups[-1].level:
            continue
        ngroups.append(p)
    return ngroups