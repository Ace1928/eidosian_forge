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
def discard_events(parameterized):
    """
    Context manager that discards any events within its scope
    triggered on the supplied parameterized object.
    """
    batch_watch = parameterized.param._BATCH_WATCH
    parameterized.param._BATCH_WATCH = True
    watchers, events = (list(parameterized.param._state_watchers), list(parameterized.param._events))
    try:
        yield
    except:
        raise
    finally:
        parameterized.param._BATCH_WATCH = batch_watch
        parameterized.param._state_watchers = watchers
        parameterized.param._events = events