from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def currency_symbols(self) -> localedata.LocaleDataDict:
    """Mapping of currency codes to symbols.

        >>> Locale('en', 'US').currency_symbols['USD']
        u'$'
        >>> Locale('es', 'CO').currency_symbols['USD']
        u'US$'
        """
    return self._data['currency_symbols']