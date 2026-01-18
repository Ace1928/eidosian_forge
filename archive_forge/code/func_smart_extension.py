from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
def smart_extension(fname, ext):
    """Append a file extension `ext` to `fname`, while keeping compressed extensions like `.bz2` or
    `.gz` (if any) at the end.

    Parameters
    ----------
    fname : str
        Filename or full path.
    ext : str
        Extension to append before any compression extensions.

    Returns
    -------
    str
        New path to file with `ext` appended.

    Examples
    --------

    .. sourcecode:: pycon

        >>> from gensim.utils import smart_extension
        >>> smart_extension("my_file.pkl.gz", ".vectors")
        'my_file.pkl.vectors.gz'

    """
    fname, oext = os.path.splitext(fname)
    if oext.endswith('.bz2'):
        fname = fname + oext[:-4] + ext + '.bz2'
    elif oext.endswith('.gz'):
        fname = fname + oext[:-3] + ext + '.gz'
    else:
        fname = fname + oext + ext
    return fname