import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
def must_skip(t):
    if not t._skipped or t._skipped.asList() == ['']:
        del t[0]
        t.pop('_skipped', None)