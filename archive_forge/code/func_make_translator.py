import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def make_translator(t):
    return functools.partial(translate_wrapper, translator=OrderedDict(t))