import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
@param.parameterized.bothmethod
def remove_diacritics(self_or_cls, identifier):
    """
        Remove diacritics and accents from the input leaving other
        unicode characters alone."""
    chars = ''
    for c in identifier:
        replacement = unicodedata.normalize('NFKD', c).encode('ASCII', 'ignore')
        if replacement != '':
            chars += bytes_to_unicode(replacement)
        else:
            chars += c
    return chars