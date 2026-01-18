from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def siParse(s, regex=FLOAT_REGEX, suffix=None):
    """Convert a value written in SI notation to a tuple (number, si_prefix, suffix).

    Example::

        siParse('100 µV")  # returns ('100', 'µ', 'V')

    Note that in the above example, the µ symbol is the "micro sign" (UTF-8
    0xC2B5), as opposed to the Greek letter mu (UTF-8 0xCEBC).

    Parameters
    ----------
    s : str
        The string to parse.
    regex : re.Pattern, optional
        Compiled regular expression object for parsing. The default is a
        general-purpose regex for parsing floating point expressions,
        potentially containing an SI prefix and a suffix.
    suffix : str, optional
        Suffix to check for in ``s``. The default (None) indicates there may or
        may not be a suffix contained in the string and it is returned if
        found. An empty string ``""`` is handled differently: if the string
        contains a suffix, it is discarded. This enables interpreting
        characters following the numerical value as an SI prefix.
    """
    s = s.strip()
    if suffix is not None and len(suffix) > 0:
        if s[-len(suffix):] != suffix:
            raise ValueError("String '%s' does not have the expected suffix '%s'" % (s, suffix))
        s = s[:-len(suffix)] + 'X'
    if suffix == '':
        s += 'X'
    m = regex.match(s)
    if m is None:
        raise ValueError('Cannot parse number "%s"' % s)
    try:
        sip = m.group('siPrefix')
    except IndexError:
        sip = ''
    if suffix is None:
        try:
            suf = m.group('suffix')
        except IndexError:
            suf = ''
    else:
        suf = suffix
    return (m.group('number'), '' if sip is None else sip, '' if suf is None else suf)