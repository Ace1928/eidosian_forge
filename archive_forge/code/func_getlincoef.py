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
def getlincoef(e, xset):
    """
    Obtain ``a`` and ``b`` when ``e == "a*x+b"``, where ``x`` is a symbol in
    xset.

    >>> getlincoef('2*x + 1', {'x'})
    (2, 1, 'x')
    >>> getlincoef('3*x + x*2 + 2 + 1', {'x'})
    (5, 3, 'x')
    >>> getlincoef('0', {'x'})
    (0, 0, None)
    >>> getlincoef('0*x', {'x'})
    (0, 0, 'x')
    >>> getlincoef('x*x', {'x'})
    (None, None, None)

    This can be tricked by sufficiently complex expressions

    >>> getlincoef('(x - 0.5)*(x - 1.5)*(x - 1)*x + 2*x + 3', {'x'})
    (2.0, 3.0, 'x')
    """
    try:
        c = int(myeval(e, {}, {}))
        return (0, c, None)
    except Exception:
        pass
    if getlincoef_re_1.match(e):
        return (1, 0, e)
    len_e = len(e)
    for x in xset:
        if len(x) > len_e:
            continue
        if re.search('\\w\\s*\\([^)]*\\b' + x + '\\b', e):
            continue
        re_1 = re.compile('(?P<before>.*?)\\b' + x + '\\b(?P<after>.*)', re.I)
        m = re_1.match(e)
        if m:
            try:
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (m1.group('before'), 0, m1.group('after'))
                    m1 = re_1.match(ee)
                b = myeval(ee, {}, {})
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (m1.group('before'), 1, m1.group('after'))
                    m1 = re_1.match(ee)
                a = myeval(ee, {}, {}) - b
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (m1.group('before'), 0.5, m1.group('after'))
                    m1 = re_1.match(ee)
                c = myeval(ee, {}, {})
                m1 = re_1.match(e)
                while m1:
                    ee = '%s(%s)%s' % (m1.group('before'), 1.5, m1.group('after'))
                    m1 = re_1.match(ee)
                c2 = myeval(ee, {}, {})
                if a * 0.5 + b == c and a * 1.5 + b == c2:
                    return (a, b, x)
            except Exception:
                pass
            break
    return (None, None, None)