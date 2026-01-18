from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
def format_argspec_plus(fn: Union[Callable[..., Any], compat.FullArgSpec], grouped: bool=True) -> Dict[str, Optional[str]]:
    """Returns a dictionary of formatted, introspected function arguments.

    A enhanced variant of inspect.formatargspec to support code generation.

    fn
       An inspectable callable or tuple of inspect getargspec() results.
    grouped
      Defaults to True; include (parens, around, argument) lists

    Returns:

    args
      Full inspect.formatargspec for fn
    self_arg
      The name of the first positional argument, varargs[0], or None
      if the function defines no positional arguments.
    apply_pos
      args, re-written in calling rather than receiving syntax.  Arguments are
      passed positionally.
    apply_kw
      Like apply_pos, except keyword-ish args are passed as keywords.
    apply_pos_proxied
      Like apply_pos but omits the self/cls argument

    Example::

      >>> format_argspec_plus(lambda self, a, b, c=3, **d: 123)
      {'grouped_args': '(self, a, b, c=3, **d)',
       'self_arg': 'self',
       'apply_kw': '(self, a, b, c=c, **d)',
       'apply_pos': '(self, a, b, c, **d)'}

    """
    if callable(fn):
        spec = compat.inspect_getfullargspec(fn)
    else:
        spec = fn
    args = compat.inspect_formatargspec(*spec)
    apply_pos = compat.inspect_formatargspec(spec[0], spec[1], spec[2], None, spec[4])
    if spec[0]:
        self_arg = spec[0][0]
        apply_pos_proxied = compat.inspect_formatargspec(spec[0][1:], spec[1], spec[2], None, spec[4])
    elif spec[1]:
        self_arg = '%s[0]' % spec[1]
        apply_pos_proxied = apply_pos
    else:
        self_arg = None
        apply_pos_proxied = apply_pos
    num_defaults = 0
    if spec[3]:
        num_defaults += len(cast(Tuple[Any], spec[3]))
    if spec[4]:
        num_defaults += len(spec[4])
    name_args = spec[0] + spec[4]
    defaulted_vals: Union[List[str], Tuple[()]]
    if num_defaults:
        defaulted_vals = name_args[0 - num_defaults:]
    else:
        defaulted_vals = ()
    apply_kw = compat.inspect_formatargspec(name_args, spec[1], spec[2], defaulted_vals, formatvalue=lambda x: '=' + str(x))
    if spec[0]:
        apply_kw_proxied = compat.inspect_formatargspec(name_args[1:], spec[1], spec[2], defaulted_vals, formatvalue=lambda x: '=' + str(x))
    else:
        apply_kw_proxied = apply_kw
    if grouped:
        return dict(grouped_args=args, self_arg=self_arg, apply_pos=apply_pos, apply_kw=apply_kw, apply_pos_proxied=apply_pos_proxied, apply_kw_proxied=apply_kw_proxied)
    else:
        return dict(grouped_args=args, self_arg=self_arg, apply_pos=apply_pos[1:-1], apply_kw=apply_kw[1:-1], apply_pos_proxied=apply_pos_proxied[1:-1], apply_kw_proxied=apply_kw_proxied[1:-1])