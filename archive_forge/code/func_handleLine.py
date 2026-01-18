from collections import OrderedDict
import functools
import itertools
import operator
import re
import sys
from pyparsing import (
import numpy
def handleLine(line):
    try:
        result = expr.parseString(line)
        return processList(result[0])
    except ParseException as exc:
        text = str(exc)
        m = re.search('(Expected .+) \\(at char (\\d+)\\), \\(line:(\\d+)', text)
        msg = m.group(1)
        if 'map|partial' in msg:
            msg = 'expected a function or operator'
        err = ExpressionError(msg)
        err.text = line
        err.offset = int(m.group(2)) + 1
        raise err