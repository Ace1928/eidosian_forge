from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def day_period_rules(self) -> localedata.LocaleDataDict:
    """Day period rules for the locale.  Used by `get_period_id`.
        """
    return self._data.get('day_period_rules', localedata.LocaleDataDict({}))