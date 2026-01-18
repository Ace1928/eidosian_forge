from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def eras(self) -> localedata.LocaleDataDict:
    """Locale display names for eras.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').eras['wide'][1]
        u'Anno Domini'
        >>> Locale('en', 'US').eras['abbreviated'][0]
        u'BC'
        """
    return self._data['eras']