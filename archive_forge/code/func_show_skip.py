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
def show_skip(t):
    if t._skipped.asList()[-1:] == ['']:
        skipped = t.pop('_skipped')
        t['_skipped'] = 'missing <' + repr(self.anchor) + '>'