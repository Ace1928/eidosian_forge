from collections import OrderedDict
import functools
import itertools
import operator
import re
import sys
from pyparsing import (
import numpy
def processList(lst):
    args = [processArg(x) for x in lst[1:]]
    func = processArg(lst[0])
    return func(*args)