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
def iter_windows(texts, window_size, copy=False, ignore_below_size=True, include_doc_num=False):
    """Produce a generator over the given texts using a sliding window of `window_size`.

    The windows produced are views of some subsequence of a text.
    To use deep copies instead, pass `copy=True`.

    Parameters
    ----------
    texts : list of str
        List of string sentences.
    window_size : int
        Size of sliding window.
    copy : bool, optional
        Produce deep copies.
    ignore_below_size : bool, optional
        Ignore documents that are not at least `window_size` in length?
    include_doc_num : bool, optional
        Yield the text position with `texts` along with each window?

    """
    for doc_num, document in enumerate(texts):
        for window in _iter_windows(document, window_size, copy, ignore_below_size):
            if include_doc_num:
                yield (doc_num, window)
            else:
                yield window