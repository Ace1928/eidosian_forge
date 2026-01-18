from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def scientific_formats(self) -> localedata.LocaleDataDict:
    """Locale patterns for scientific number formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').scientific_formats[None]
        <NumberPattern u'#E0'>
        """
    return self._data['scientific_formats']