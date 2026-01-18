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
def clip_parents(self, i):
    """Return list of parent clip paths.

                Args:
                    i: (int) return parents of this path.
                Returns:
                    List of the clip parents."""
    if i >= self.path_count:
        raise IndexError('bad path index')
    while i < 0:
        i += self.path_count
    lvl = self.paths[i].level
    clips = list(reversed([p for p in self.paths[:i] if p.type == 'clip' and p.level < lvl]))
    if clips == []:
        return []
    nclips = [clips[0]]
    for p in clips[1:]:
        if p.level >= nclips[-1].level:
            continue
        nclips.append(p)
    return nclips