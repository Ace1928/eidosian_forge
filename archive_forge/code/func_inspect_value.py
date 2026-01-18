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
def inspect_value(self_, name):
    """
        Return the current value of the named attribute without modifying it.

        Same as getattr() except for Dynamic parameters, which have their
        last generated value returned.
        """
    cls_or_slf = self_.self_or_cls
    param_obj = cls_or_slf.param.objects('existing').get(name)
    if not param_obj:
        value = getattr(cls_or_slf, name)
    elif hasattr(param_obj, 'attribs'):
        value = [cls_or_slf.param.inspect_value(a) for a in param_obj.attribs]
    elif not hasattr(param_obj, '_inspect'):
        value = getattr(cls_or_slf, name)
    elif isinstance(cls_or_slf, type):
        value = param_obj._inspect(None, cls_or_slf)
    else:
        value = param_obj._inspect(cls_or_slf, None)
    return value