from collections import OrderedDict
import functools
import itertools
import operator
import re
import sys
from pyparsing import (
import numpy
def resolve_var(s, l, t):
    try:
        return _ctx.get(t[0])
    except KeyError:
        err = ExpressionError("name '%s' is not defined" % t[0])
        err.text = s
        err.offset = l + 1
        raise err