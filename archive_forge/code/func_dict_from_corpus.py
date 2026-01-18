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
def dict_from_corpus(corpus):
    """Scan corpus for all word ids that appear in it, then construct a mapping
    which maps each `word_id` -> `str(word_id)`.

    Parameters
    ----------
    corpus : iterable of iterable of (int, numeric)
        Collection of texts in BoW format.

    Returns
    ------
    id2word : :class:`~gensim.utils.FakeDict`
        "Fake" mapping which maps each `word_id` -> `str(word_id)`.

    Warnings
    --------
    This function is used whenever *words* need to be displayed (as opposed to just their ids)
    but no `word_id` -> `word` mapping was provided. The resulting mapping only covers words actually
    used in the corpus, up to the highest `word_id` found.

    """
    num_terms = 1 + get_max_id(corpus)
    id2word = FakeDict(num_terms)
    return id2word