import inspect
import locale
import os
import pydoc
import re
import textwrap
import warnings
from collections import defaultdict, deque
from itertools import chain, combinations, islice, tee
from pprint import pprint
from urllib.request import (
from nltk.collections import *
from nltk.internals import deprecated, raise_unorderable_types, slice_bounds
def filestring(f):
    if hasattr(f, 'read'):
        return f.read()
    elif isinstance(f, str):
        with open(f) as infile:
            return infile.read()
    else:
        raise ValueError('Must be called with a filename or file-like object')