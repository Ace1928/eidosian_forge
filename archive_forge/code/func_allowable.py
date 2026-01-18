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
def allowable(self_or_cls, name, disable_leading_underscore=None):
    disabled_reprs = ['javascript', 'jpeg', 'json', 'latex', 'latex', 'pdf', 'png', 'svg', 'markdown']
    disabled_ = self_or_cls.disable_leading_underscore if disable_leading_underscore is None else disable_leading_underscore
    if disabled_ and name.startswith('_'):
        return False
    isrepr = any((f'_repr_{el}_' == name for el in disabled_reprs))
    return name not in self_or_cls.disallowed and (not isrepr)