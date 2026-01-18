from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def territories(self) -> localedata.LocaleDataDict:
    """Mapping of script codes to translated script names.

        >>> Locale('es', 'CO').territories['DE']
        u'Alemania'

        See `ISO 3166 <http://www.iso.org/iso/en/prods-services/iso3166ma/>`_
        for more information.
        """
    return self._data['territories']