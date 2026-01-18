import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
@_deprecated(extra_msg='Use instead `for k,v in p.param.objects().items(): print(f"{p.__class__.name}.{k}={repr(v.default)}")`')
def print_param_defaults(self_):
    """Print the default values of all cls's Parameters.

        .. deprecated:: 1.12.0
            Use instead `for k,v in p.param.objects().items(): print(f"{p.__class__.name}.{k}={repr(v.default)}")`
        """
    cls = self_.cls
    for key, val in cls.__dict__.items():
        if isinstance(val, Parameter):
            print(cls.__name__ + '.' + key + '=' + repr(val.default))