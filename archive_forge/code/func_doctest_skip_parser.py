import os
import re
import struct
import threading
import functools
import inspect
from tempfile import NamedTemporaryFile
import numpy as np
from numpy import testing
from numpy.testing import (
from .. import data, io
from ..data._fetchers import _fetch
from ..util import img_as_uint, img_as_float, img_as_int, img_as_ubyte
from ._warnings import expected_warnings
import pytest
def doctest_skip_parser(func):
    """Decorator replaces custom skip test markup in doctests

    Say a function has a docstring::

        >>> something, HAVE_AMODULE, HAVE_BMODULE = 0, False, False
        >>> something # skip if not HAVE_AMODULE
        0
        >>> something # skip if HAVE_BMODULE
        0

    This decorator will evaluate the expression after ``skip if``.  If this
    evaluates to True, then the comment is replaced by ``# doctest: +SKIP``. If
    False, then the comment is just removed. The expression is evaluated in the
    ``globals`` scope of `func`.

    For example, if the module global ``HAVE_AMODULE`` is False, and module
    global ``HAVE_BMODULE`` is False, the returned function will have docstring::

        >>> something # doctest: +SKIP
        >>> something + else # doctest: +SKIP
        >>> something # doctest: +SKIP

    """
    lines = func.__doc__.split('\n')
    new_lines = []
    for line in lines:
        match = SKIP_RE.match(line)
        if match is None:
            new_lines.append(line)
            continue
        code, space, expr = match.groups()
        try:
            if eval(expr, func.__globals__):
                code = code + space + '# doctest: +SKIP'
        except AttributeError:
            if eval(expr, func.__init__.__globals__):
                code = code + space + '# doctest: +SKIP'
        new_lines.append(code)
    func.__doc__ = '\n'.join(new_lines)
    return func