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
@classmethod
def compare_mapping(cls, obj1, obj2):
    if type(obj1) != type(obj2) or len(obj1) != len(obj2):
        return False
    for k in obj1:
        if k in obj2:
            if not cls.is_equal(obj1[k], obj2[k]):
                return False
        else:
            return False
    return True