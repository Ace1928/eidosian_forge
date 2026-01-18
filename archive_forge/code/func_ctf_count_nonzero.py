import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def ctf_count_nonzero(x):
    return (x != 0).astype(int).sum()