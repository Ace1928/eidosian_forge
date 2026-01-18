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
def deaccent(text):
    """Remove letter accents from the given string.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Unicode string without accents.

    Examples
    --------
    .. sourcecode:: pycon

        >>> from gensim.utils import deaccent
        >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
        u'Sef chomutovskych komunistu dostal postou bily prasek'

    """
    if not isinstance(text, str):
        text = text.decode('utf8')
    norm = unicodedata.normalize('NFD', text)
    result = ''.join((ch for ch in norm if unicodedata.category(ch) != 'Mn'))
    return unicodedata.normalize('NFC', result)