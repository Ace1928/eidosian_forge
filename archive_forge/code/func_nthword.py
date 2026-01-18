from __future__ import absolute_import, print_function, division
from petl.util.base import values, header, Table
def nthword(n, sep=None):
    """
    Construct a function to return the nth word in a string. E.g.::

        >>> import petl as etl
        >>> s = 'foo bar'
        >>> f = etl.nthword(0)
        >>> f(s)
        'foo'
        >>> g = etl.nthword(1)
        >>> g(s)
        'bar'

    Intended for use with :func:`petl.transform.conversions.convert`.

    """
    return lambda s: s.split(sep)[n]