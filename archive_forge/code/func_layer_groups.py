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
def layer_groups(ordering, length=2):
    """
   Splits a global ordering of Layers into groups based on a slice of
   the spec.  The grouping behavior can be modified by changing the
   length of spec the entries are grouped by.
   """
    group_orderings = defaultdict(list)
    for el in ordering:
        group_orderings[el[:length]].append(el)
    return group_orderings