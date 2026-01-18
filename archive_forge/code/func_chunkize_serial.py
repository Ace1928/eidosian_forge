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
def chunkize_serial(iterable, chunksize, as_numpy=False, dtype=np.float32):
    """Yield elements from `iterable` in "chunksize"-ed groups.

    The last returned element may be smaller if the length of collection is not divisible by `chunksize`.

    Parameters
    ----------
    iterable : iterable of object
        An iterable.
    chunksize : int
        Split iterable into chunks of this size.
    as_numpy : bool, optional
        Yield chunks as `np.ndarray` instead of lists.

    Yields
    ------
    list OR np.ndarray
        "chunksize"-ed chunks of elements from `iterable`.

    Examples
    --------
    .. sourcecode:: pycon

        >>> print(list(grouper(range(10), 3)))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    """
    it = iter(iterable)
    while True:
        if as_numpy:
            wrapped_chunk = [[np.array(doc, dtype=dtype) for doc in itertools.islice(it, int(chunksize))]]
        else:
            wrapped_chunk = [list(itertools.islice(it, int(chunksize)))]
        if not wrapped_chunk[0]:
            break
        yield wrapped_chunk.pop()