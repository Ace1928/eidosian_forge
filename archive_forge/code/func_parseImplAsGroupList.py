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
def parseImplAsGroupList(self, instring, loc, doActions=True):
    result = self.re_match(instring, loc)
    if not result:
        raise ParseException(instring, loc, self.errmsg, self)
    loc = result.end()
    ret = result.groups()
    return (loc, ret)