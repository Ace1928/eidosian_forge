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
def prune_vocab(vocab, min_reduce, trim_rule=None):
    """Remove all entries from the `vocab` dictionary with count smaller than `min_reduce`.

    Modifies `vocab` in place, returns the sum of all counts that were pruned.

    Parameters
    ----------
    vocab : dict
        Input dictionary.
    min_reduce : int
        Frequency threshold for tokens in `vocab`.
    trim_rule : function, optional
        Function for trimming entities from vocab, default behaviour is `vocab[w] <= min_reduce`.

    Returns
    -------
    result : int
        Sum of all counts that were pruned.

    """
    result = 0
    old_len = len(vocab)
    for w in list(vocab):
        if not keep_vocab_item(w, vocab[w], min_reduce, trim_rule):
            result += vocab[w]
            del vocab[w]
    logger.info('pruned out %i tokens with count <=%i (before %i, after %i)', old_len - len(vocab), min_reduce, old_len, len(vocab))
    return result