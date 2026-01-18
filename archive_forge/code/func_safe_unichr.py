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
def safe_unichr(intval):
    """Create a unicode character from its integer value. In case `unichr` fails, render the character
    as an escaped `\\U<8-byte hex value of intval>` string.

    Parameters
    ----------
    intval : int
        Integer code of character

    Returns
    -------
    string
        Unicode string of character

    """
    try:
        return chr(intval)
    except ValueError:
        s = '\\U%08x' % intval
        return s.decode('unicode-escape')