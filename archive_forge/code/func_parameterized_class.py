import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
def parameterized_class(name, params, bases=Parameterized):
    """
    Dynamically create a parameterized class with the given name and the
    supplied parameters, inheriting from the specified base(s).
    """
    if not (isinstance(bases, list) or isinstance(bases, tuple)):
        bases = [bases]
    return type(name, tuple(bases), params)