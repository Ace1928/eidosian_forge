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
def instance_descriptor(f):

    def _f(self, obj, val):
        if obj is not None:
            instance_param = obj._param__private.params.get(self.name)
            if instance_param is None:
                instance_param = _instantiated_parameter(obj, self)
            if instance_param is not None and self is not instance_param:
                instance_param.__set__(obj, val)
                return
        return f(self, obj, val)
    return _f