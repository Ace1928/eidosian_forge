import builtins
import datetime as dt
import hashlib
import inspect
import itertools
import json
import numbers
import operator
import pickle
import string
import sys
import time
import types
import unicodedata
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
def resolve_dependent_value(value):
    """Resolves parameter dependencies on the supplied value

    Resolves parameter values, Parameterized instance methods,
    parameterized functions with dependencies on the supplied value,
    including such parameters embedded in a list, tuple, dictionary, or slice.

    Args:
       value: A value which will be resolved

    Returns:
       A new value where any parameter dependencies have been
       resolved.
    """
    range_widget = False
    if isinstance(value, list):
        value = [resolve_dependent_value(v) for v in value]
    elif isinstance(value, tuple):
        value = tuple((resolve_dependent_value(v) for v in value))
    elif isinstance(value, dict):
        value = {resolve_dependent_value(k): resolve_dependent_value(v) for k, v in value.items()}
    elif isinstance(value, slice):
        value = slice(resolve_dependent_value(value.start), resolve_dependent_value(value.stop), resolve_dependent_value(value.step))
    if 'panel' in sys.modules:
        from panel.depends import param_value_if_widget
        from panel.widgets import RangeSlider
        range_widget = isinstance(value, RangeSlider)
        if param_version > Version('2.0.0rc1'):
            value = param.parameterized.resolve_value(value)
        else:
            value = param_value_if_widget(value)
    if is_param_method(value, has_deps=True):
        value = value()
    elif isinstance(value, param.Parameter) and isinstance(value.owner, param.Parameterized):
        value = getattr(value.owner, value.name)
    elif isinstance(value, FunctionType) and hasattr(value, '_dinfo'):
        deps = value._dinfo
        args = (getattr(p.owner, p.name) for p in deps.get('dependencies', []))
        kwargs = {k: getattr(p.owner, p.name) for k, p in deps.get('kw', {}).items()}
        value = value(*args, **kwargs)
    if isinstance(value, tuple) and range_widget:
        value = slice(*value)
    return value