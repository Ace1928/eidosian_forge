from __future__ import division, absolute_import, print_function
from itertools import chain, starmap
import itertools       # since zip_longest doesn't exist on Py2
from past.types import basestring
from past.utils import PY3
def oldfilter(*args):
    """
        filter(function or None, sequence) -> list, tuple, or string

        Return those items of sequence for which function(item) is true.
        If function is None, return the items that are true.  If sequence
        is a tuple or string, return the same type, else return a list.
        """
    mytype = type(args[1])
    if isinstance(args[1], basestring):
        return mytype().join(builtins.filter(*args))
    elif isinstance(args[1], (tuple, list)):
        return mytype(builtins.filter(*args))
    else:
        return list(builtins.filter(*args))