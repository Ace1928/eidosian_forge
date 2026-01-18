from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def periods(self) -> localedata.LocaleDataDict:
    """Locale display names for day periods (AM/PM).

        >>> Locale('en', 'US').periods['am']
        u'AM'
        """
    try:
        return self._data['day_periods']['stand-alone']['wide']
    except KeyError:
        return localedata.LocaleDataDict({})