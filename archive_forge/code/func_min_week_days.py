from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def min_week_days(self) -> int:
    """The minimum number of days in a week so that the week is counted as
        the first week of a year or month.

        >>> Locale('de', 'DE').min_week_days
        4
        """
    return self._data['week_data']['min_days']