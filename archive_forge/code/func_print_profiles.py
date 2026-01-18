import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
def print_profiles(self, w=0, cutoff=0, **print3options):
    """Print the profiles above *cutoff* percentage.

        The available options and defaults are:

             *w=0*           -- indentation for each line

             *cutoff=0*      -- minimum percentage printed

             *print3options* -- some keyword arguments, like Python 3+ print
        """
    t = [(v, k) for k, v in _items(self._profs) if v.total > 0 or v.number > 1]
    if len(self._profs) - len(t) < 9:
        t = [(v, k) for k, v in _items(self._profs)]
    if t:
        s = _NN
        if self._total:
            s = ' (% of grand total)'
            c = int(cutoff) if cutoff else self._cutoff_
            C = int(c * 0.01 * self._total)
        else:
            C = c = 0
        self._printf('%s%*d profile%s:  total%s, average, and largest flat size%s:  largest object', linesep, w, len(t), _plural(len(t)), s, self._incl, **print3options)
        r = len(t)
        t = [(v, self._prepr(k)) for v, k in t]
        for v, k in sorted(t, reverse=True):
            s = 'object%(plural)s:  %(total)s, %(avg)s, %(high)s:  %(obj)s%(lengstr)s' % v.format(self._clip_, self._total)
            self._printf('%*d %s %s', w, v.number, k, s, **print3options)
            r -= 1
            if r > 1 and v.total < C:
                self._printf('%+*d profiles below cutoff (%.0f%%)', w, r, c)
                break
        z = len(self._profs) - len(t)
        if z > 0:
            self._printf('%+*d %r object%s', w, z, 'zero', _plural(z), **print3options)