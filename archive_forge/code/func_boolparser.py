from __future__ import absolute_import, print_function, division
import datetime
from petl.compat import long
def boolparser(true_strings=('true', 't', 'yes', 'y', '1'), false_strings=('false', 'f', 'no', 'n', '0'), case_sensitive=False, strict=True):
    """Return a function to parse strings as :class:`bool` objects using a
    given set of string representations for `True` and `False`. E.g.::

        >>> from petl import boolparser
        >>> mybool = boolparser(true_strings=['yes', 'y'], false_strings=['no', 'n'])
        >>> mybool('y')
        True
        >>> mybool('yes')
        True
        >>> mybool('Y')
        True
        >>> mybool('No')
        False
        >>> try:
        ...     mybool('foo')
        ... except ValueError as e:
        ...     print(e)
        ...
        value is not one of recognised boolean strings: 'foo'
        >>> try:
        ...     mybool('True')
        ... except ValueError as e:
        ...     print(e)
        ...
        value is not one of recognised boolean strings: 'true'

    If ``strict=False`` then if an error occurs when parsing, the original
    value will be returned as-is, and no error will be raised.

    """
    if not case_sensitive:
        true_strings = [s.lower() for s in true_strings]
        false_strings = [s.lower() for s in false_strings]

    def parser(value):
        value = value.strip()
        if not case_sensitive:
            value = value.lower()
        if value in true_strings:
            return True
        elif value in false_strings:
            return False
        elif strict:
            raise ValueError('value is not one of recognised boolean strings: %r' % value)
        else:
            return value
    return parser