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
def formatannotationrelativeto(object):
    module = getattr(object, '__module__', None)

    def _formatannotation(annotation):
        return formatannotation(annotation, module)
    return _formatannotation