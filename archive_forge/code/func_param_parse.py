import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def param_parse(d, params):
    """Recursively parse array dimensions.

    Parses the declaration of an array variable or parameter
    `dimension` keyword, and is called recursively if the
    dimension for this array is a previously defined parameter
    (found in `params`).

    Parameters
    ----------
    d : str
        Fortran expression describing the dimension of an array.
    params : dict
        Previously parsed parameters declared in the Fortran source file.

    Returns
    -------
    out : str
        Parsed dimension expression.

    Examples
    --------

    * If the line being analyzed is

      `integer, parameter, dimension(2) :: pa = (/ 3, 5 /)`

      then `d = 2` and we return immediately, with

    >>> d = '2'
    >>> param_parse(d, params)
    2

    * If the line being analyzed is

      `integer, parameter, dimension(pa) :: pb = (/1, 2, 3/)`

      then `d = 'pa'`; since `pa` is a previously parsed parameter,
      and `pa = 3`, we call `param_parse` recursively, to obtain

    >>> d = 'pa'
    >>> params = {'pa': 3}
    >>> param_parse(d, params)
    3

    * If the line being analyzed is

      `integer, parameter, dimension(pa(1)) :: pb = (/1, 2, 3/)`

      then `d = 'pa(1)'`; since `pa` is a previously parsed parameter,
      and `pa(1) = 3`, we call `param_parse` recursively, to obtain

    >>> d = 'pa(1)'
    >>> params = dict(pa={1: 3, 2: 5})
    >>> param_parse(d, params)
    3
    """
    if '(' in d:
        dname = d[:d.find('(')]
        ddims = d[d.find('(') + 1:d.rfind(')')]
        index = int(param_parse(ddims, params))
        return str(params[dname][index])
    elif d in params:
        return str(params[d])
    else:
        for p in params:
            re_1 = re.compile('(?P<before>.*?)\\b' + p + '\\b(?P<after>.*)', re.I)
            m = re_1.match(d)
            while m:
                d = m.group('before') + str(params[p]) + m.group('after')
                m = re_1.match(d)
        return d