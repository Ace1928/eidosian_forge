from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def day_periods(self) -> localedata.LocaleDataDict:
    """Locale display names for various day periods (not necessarily only AM/PM).

        These are not meant to be used without the relevant `day_period_rules`.
        """
    return self._data['day_periods']