from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
def get_territory_name(self, locale: Locale | str | None=None) -> str | None:
    """Return the territory name in the given locale."""
    if locale is None:
        locale = self
    locale = Locale.parse(locale)
    return locale.territories.get(self.territory or '')