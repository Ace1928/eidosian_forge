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
def sanitize_py3(self, name):
    if not name.isidentifier():
        return '_'.join(self.sanitize(name, lambda c: ('_' + c).isidentifier()))
    else:
        return name