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
def get_value_generator(self_, name):
    """
        Return the value or value-generating object of the named
        attribute.

        For most parameters, this is simply the parameter's value
        (i.e. the same as getattr()), but Dynamic parameters have
        their value-generating object returned.
        """
    cls_or_slf = self_.self_or_cls
    param_obj = cls_or_slf.param.objects('existing').get(name)
    if not param_obj:
        value = getattr(cls_or_slf, name)
    elif hasattr(param_obj, 'attribs'):
        value = [cls_or_slf.param.get_value_generator(a) for a in param_obj.attribs]
    elif not hasattr(param_obj, '_value_is_dynamic'):
        value = getattr(cls_or_slf, name)
    elif isinstance(cls_or_slf, Parameterized) and name in cls_or_slf._param__private.values:
        value = cls_or_slf._param__private.values[name]
    else:
        value = param_obj.default
    return value