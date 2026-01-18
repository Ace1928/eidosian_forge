from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def number_symbols(self) -> localedata.LocaleDataDict:
    """Symbols used in number formatting by number system.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('fr', 'FR').number_symbols["latn"]['decimal']
        u','
        >>> Locale('fa', 'IR').number_symbols["arabext"]['decimal']
        u'٫'
        >>> Locale('fa', 'IR').number_symbols["latn"]['decimal']
        u'.'
        """
    return self._data['number_symbols']