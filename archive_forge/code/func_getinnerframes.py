import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import functools
import builtins
from keyword import iskeyword
from operator import attrgetter
from collections import namedtuple, OrderedDict
def getinnerframes(tb, context=1):
    """Get a list of records for a traceback's frame and all lower frames.

    Each record contains a frame object, filename, line number, function
    name, a list of lines of context, and index within the context."""
    framelist = []
    while tb:
        traceback_info = getframeinfo(tb, context)
        frameinfo = (tb.tb_frame,) + traceback_info
        framelist.append(FrameInfo(*frameinfo, positions=traceback_info.positions))
        tb = tb.tb_next
    return framelist