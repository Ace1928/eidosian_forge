import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def getfullargspec_no_self(func):
    """inspect.getfullargspec replacement using inspect.signature.

    If func is a bound method, do not list the 'self' parameter.

    Parameters
    ----------
    func : callable
        A callable to inspect

    Returns
    -------
    fullargspec : FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                              kwonlydefaults, annotations)

        NOTE: if the first argument of `func` is self, it is *not*, I repeat
        *not*, included in fullargspec.args.
        This is done for consistency between inspect.getargspec() under
        Python 2.x, and inspect.signature() under Python 3.x.

    """
    sig = inspect.signature(func)
    args = [p.name for p in sig.parameters.values() if p.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]]
    varargs = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_POSITIONAL]
    varargs = varargs[0] if varargs else None
    varkw = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD]
    varkw = varkw[0] if varkw else None
    defaults = tuple((p.default for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is not p.empty)) or None
    kwonlyargs = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.KEYWORD_ONLY]
    kwdefaults = {p.name: p.default for p in sig.parameters.values() if p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is not p.empty}
    annotations = {p.name: p.annotation for p in sig.parameters.values() if p.annotation is not p.empty}
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwdefaults or None, annotations)