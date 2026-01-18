from collections import OrderedDict
import functools
import itertools
import operator
import re
import sys
from pyparsing import (
import numpy
def resolve_func(s, l, t):
    try:
        return func_map[t[0]] if t[0] in func_map else getattr(numpy, t[0])
    except AttributeError:
        err = ExpressionError("'%s' is not a function or operator" % t[0])
        err.text = s
        err.offset = l + 1
        raise err