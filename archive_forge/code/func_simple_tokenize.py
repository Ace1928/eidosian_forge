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
def simple_tokenize(text):
    """Tokenize input test using :const:`gensim.utils.PAT_ALPHABETIC`.

    Parameters
    ----------
    text : str
        Input text.

    Yields
    ------
    str
        Tokens from `text`.

    """
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()