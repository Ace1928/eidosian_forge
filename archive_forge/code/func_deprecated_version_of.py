import os
import subprocess as sp
import sys
import warnings
import proglog
from .compat import DEVNULL
def deprecated_version_of(f, oldname, newname=None):
    """ Indicates that a function is deprecated and has a new name.

    `f` is the new function, `oldname` the name of the deprecated
    function, `newname` the name of `f`, which can be automatically
    found.

    Returns
    ========

    f_deprecated
      A function that does the same thing as f, but with a docstring
      and a printed message on call which say that the function is
      deprecated and that you should use f instead.

    Examples
    =========

    >>> # The badly named method 'to_file' is replaced by 'write_file'
    >>> class Clip:
    >>>    def write_file(self, some args):
    >>>        # blablabla
    >>>
    >>> Clip.to_file = deprecated_version_of(Clip.write_file, 'to_file')
    """
    if newname is None:
        newname = f.__name__
    warning = 'The function ``%s`` is deprecated and is kept temporarily for backwards compatibility.\nPlease use the new name, ``%s``, instead.' % (oldname, newname)

    def fdepr(*a, **kw):
        warnings.warn('MoviePy: ' + warning, PendingDeprecationWarning)
        return f(*a, **kw)
    fdepr.__doc__ = warning
    return fdepr