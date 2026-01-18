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
def get_max_id(corpus):
    """Get the highest feature id that appears in the corpus.

    Parameters
    ----------
    corpus : iterable of iterable of (int, numeric)
        Collection of texts in BoW format.

    Returns
    ------
    int
        Highest feature id.

    Notes
    -----
    For empty `corpus` return -1.

    """
    maxid = -1
    for document in corpus:
        if document:
            maxid = max(maxid, max((fieldid for fieldid, _ in document)))
    return maxid