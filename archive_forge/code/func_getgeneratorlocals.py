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
def getgeneratorlocals(generator):
    """
    Get the mapping of generator local variables to their current values.

    A dict is returned, with the keys the local variable names and values the
    bound values."""
    if not isgenerator(generator):
        raise TypeError('{!r} is not a Python generator'.format(generator))
    frame = getattr(generator, 'gi_frame', None)
    if frame is not None:
        return generator.gi_frame.f_locals
    else:
        return {}