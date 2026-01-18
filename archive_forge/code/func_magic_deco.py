import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def magic_deco(arg):
    caller = sys._getframe(1)
    for ns in ['f_locals', 'f_globals', 'f_builtins']:
        get_ipython = getattr(caller, ns).get('get_ipython')
        if get_ipython is not None:
            break
    else:
        raise NameError('Decorator can only run in context where `get_ipython` exists')
    ip = get_ipython()
    if callable(arg):
        func = arg
        name = func.__name__
        ip.register_magic_function(func, magic_kind, name)
        retval = arg
    elif isinstance(arg, str):
        name = arg

        def mark(func, *a, **kw):
            ip.register_magic_function(func, magic_kind, name)
            return func
        retval = mark
    else:
        raise TypeError('Decorator can only be called with string or function')
    return retval