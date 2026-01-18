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
def strided_windows(ndarray, window_size):
    """Produce a numpy.ndarray of windows, as from a sliding window.

    Parameters
    ----------
    ndarray : numpy.ndarray
        Input array
    window_size : int
        Sliding window size.

    Returns
    -------
    numpy.ndarray
        Subsequences produced by sliding a window of the given size over the `ndarray`.
        Since this uses striding, the individual arrays are views rather than copies of `ndarray`.
        Changes to one view modifies the others and the original.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import strided_windows
        >>> strided_windows(np.arange(5), 2)
        array([[0, 1],
               [1, 2],
               [2, 3],
               [3, 4]])
        >>> strided_windows(np.arange(10), 5)
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8],
               [5, 6, 7, 8, 9]])

    """
    ndarray = np.asarray(ndarray)
    if window_size == ndarray.shape[0]:
        return np.array([ndarray])
    elif window_size > ndarray.shape[0]:
        return np.ndarray((0, 0))
    stride = ndarray.strides[0]
    return np.lib.stride_tricks.as_strided(ndarray, shape=(ndarray.shape[0] - window_size + 1, window_size), strides=(stride, stride))