import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def cvt_fn(s, l, t):
    try:
        return datetime.strptime(t[0], fmt)
    except ValueError as ve:
        raise ParseException(s, l, str(ve))