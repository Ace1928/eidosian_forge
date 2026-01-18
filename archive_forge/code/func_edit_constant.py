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
@contextmanager
def edit_constant(parameterized):
    """
    Temporarily set parameters on Parameterized object to constant=False
    to allow editing them.
    """
    params = parameterized.param.objects('existing').values()
    constants = [p.constant for p in params]
    for p in params:
        p.constant = False
    try:
        yield
    except:
        raise
    finally:
        for p, const in zip(params, constants):
            p.constant = const