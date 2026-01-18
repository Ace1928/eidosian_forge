from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def interval_formats(self) -> localedata.LocaleDataDict:
    """Locale patterns for interval formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        How to format date intervals in Finnish when the day is the
        smallest changing component:

        >>> Locale('fi_FI').interval_formats['MEd']['d']
        [u'E d.\u2009–\u2009', u'E d.M.']

        .. seealso::

           The primary API to use this data is :py:func:`babel.dates.format_interval`.


        :rtype: dict[str, dict[str, list[str]]]
        """
    return self._data['interval_formats']