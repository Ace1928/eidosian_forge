import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
@_deprecated()
def produce_value(value_obj):
    """
    A helper function that produces an actual parameter from a stored
    object: if the object is callable, call it, otherwise return the
    object.

    .. deprecated:: 2.0.0
    """
    return _produce_value(value_obj)