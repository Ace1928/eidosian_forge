from __future__ import absolute_import, print_function, division
import datetime
from petl.compat import long
def dateparser(fmt, strict=True):
    """Return a function to parse strings as :class:`datetime.date` objects
    using a given format. E.g.::

        >>> from petl import dateparser
        >>> isodate = dateparser('%Y-%m-%d')
        >>> isodate('2002-12-25')
        datetime.date(2002, 12, 25)
        >>> try:
        ...     isodate('2002-02-30')
        ... except ValueError as e:
        ...     print(e)
        ...
        day is out of range for month

    If ``strict=False`` then if an error occurs when parsing, the original
    value will be returned as-is, and no error will be raised.

    """

    def parser(value):
        try:
            return datetime.datetime.strptime(value.strip(), fmt).date()
        except Exception as e:
            if strict:
                raise e
            else:
                return value
    return parser