from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def datetime_skeletons(self) -> localedata.LocaleDataDict:
    """Locale patterns for formatting parts of a datetime.

        >>> Locale('en').datetime_skeletons['MEd']
        <DateTimePattern u'E, M/d'>
        >>> Locale('fr').datetime_skeletons['MEd']
        <DateTimePattern u'E dd/MM'>
        >>> Locale('fr').datetime_skeletons['H']
        <DateTimePattern u"HH 'h'">
        """
    return self._data['datetime_skeletons']