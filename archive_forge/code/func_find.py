import contextlib
import os
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.sexp as sexp
from rpy2.robjects.robject import RObjectMixin
from rpy2.robjects import conversion
def find(self, item: str, wantfun: bool=False):
    """Find an item, starting with this R environment.

        Raises a `KeyError` if the key cannot be found.

        This method is called `find` because it is somewhat different
        from the method :meth:`get` in Python mappings such :class:`dict`.
        This is looking for a key across enclosing environments, returning
        the first key found.

        :param item: string (name/symbol)
        :rtype: object (as returned by :func:`conversion.converter.rpy2py`)
        """
    res = super(Environment, self).find(item, wantfun=wantfun)
    res = conversion.get_conversion().rpy2py(res)
    try:
        res.__rname__ = item
    except AttributeError:
        pass
    return res