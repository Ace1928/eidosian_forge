import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
def contiguous_regions(mask):
    """
    Return a list of (ind0, ind1) such that ``mask[ind0:ind1].all()`` is
    True and we cover all such regions.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.size:
        return []
    idx, = np.nonzero(mask[:-1] != mask[1:])
    idx += 1
    idx = idx.tolist()
    if mask[0]:
        idx = [0] + idx
    if mask[-1]:
        idx.append(len(mask))
    return list(zip(idx[::2], idx[1::2]))