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
def keep_vocab_item(word, count, min_count, trim_rule=None):
    """Should we keep `word` in the vocab or remove it?

    Parameters
    ----------
    word : str
        Input word.
    count : int
        Number of times that word appeared in a corpus.
    min_count : int
        Discard words with frequency smaller than this.
    trim_rule : function, optional
        Custom function to decide whether to keep or discard this word.
        If a custom `trim_rule` is not specified, the default behaviour is simply `count >= min_count`.

    Returns
    -------
    bool
        True if `word` should stay, False otherwise.

    """
    default_res = count >= min_count
    if trim_rule is None:
        return default_res
    else:
        rule_res = trim_rule(word, count, min_count)
        if rule_res == RULE_KEEP:
            return True
        elif rule_res == RULE_DISCARD:
            return False
        else:
            return default_res