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
def shortened_character_name(self_or_cls, c, eliminations=None, substitutions=None, transforms=None):
    """
        Given a unicode character c, return the shortened unicode name
        (as a list of tokens) by applying the eliminations,
        substitutions and transforms.
        """
    if transforms is None:
        transforms = []
    if substitutions is None:
        substitutions = {}
    if eliminations is None:
        eliminations = []
    name = unicodedata.name(c).lower()
    for elim in eliminations:
        name = name.replace(elim, '')
    for i, o in substitutions.items():
        name = name.replace(i, o)
    for transform in transforms:
        name = transform(name)
    return ' '.join(name.strip().split()).replace(' ', '_').replace('-', '_')