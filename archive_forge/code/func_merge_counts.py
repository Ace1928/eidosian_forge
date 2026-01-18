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
def merge_counts(dict1, dict2):
    """Merge `dict1` of (word, freq1) and `dict2` of (word, freq2) into `dict1` of (word, freq1+freq2).
    Parameters
    ----------
    dict1 : dict of (str, int)
        First dictionary.
    dict2 : dict of (str, int)
        Second dictionary.
    Returns
    -------
    result : dict
        Merged dictionary with sum of frequencies as values.
    """
    for word, freq in dict2.items():
        if word in dict1:
            dict1[word] += freq
        else:
            dict1[word] = freq
    return dict1