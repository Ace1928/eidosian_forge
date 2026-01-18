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
def revdict(d):
    """Reverse a dictionary mapping, i.e. `{1: 2, 3: 4}` -> `{2: 1, 4: 3}`.

    Parameters
    ----------
    d : dict
        Input dictionary.

    Returns
    -------
    dict
        Reversed dictionary mapping.

    Notes
    -----
    When two keys map to the same value, only one of them will be kept in the result (which one is kept is arbitrary).

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import revdict
        >>> d = {1: 2, 3: 4}
        >>> revdict(d)
        {2: 1, 4: 3}

    """
    return {v: k for k, v in dict(d).items()}