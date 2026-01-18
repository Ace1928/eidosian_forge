from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def exec_(_code_, _globs_=None, _locs_=None):
    """Execute code in a namespace."""
    if _globs_ is None:
        frame = sys._getframe(1)
        _globs_ = frame.f_globals
        if _locs_ is None:
            _locs_ = frame.f_locals
        del frame
    elif _locs_ is None:
        _locs_ = _globs_
    exec('exec _code_ in _globs_, _locs_')