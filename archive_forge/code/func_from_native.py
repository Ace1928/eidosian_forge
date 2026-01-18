from __future__ import annotations
import re
from typing import Any, Generic, Pattern, Type, TypeVar, Union
from bson._helpers import _getstate_slots, _setstate_slots
from bson.son import RE_TYPE
@classmethod
def from_native(cls: Type[Regex[Any]], regex: Pattern[_T]) -> Regex[_T]:
    """Convert a Python regular expression into a ``Regex`` instance.

        Note that in Python 3, a regular expression compiled from a
        :class:`str` has the ``re.UNICODE`` flag set. If it is undesirable
        to store this flag in a BSON regular expression, unset it first::

          >>> pattern = re.compile('.*')
          >>> regex = Regex.from_native(pattern)
          >>> regex.flags ^= re.UNICODE
          >>> db.collection.insert_one({'pattern': regex})

        :Parameters:
          - `regex`: A regular expression object from ``re.compile()``.

        .. warning::
           Python regular expressions use a different syntax and different
           set of flags than MongoDB, which uses `PCRE`_. A regular
           expression retrieved from the server may not compile in
           Python, or may match a different set of strings in Python than
           when used in a MongoDB query.

        .. _PCRE: http://www.pcre.org/
        """
    if not isinstance(regex, RE_TYPE):
        raise TypeError('regex must be a compiled regular expression, not %s' % type(regex))
    return Regex(regex.pattern, regex.flags)