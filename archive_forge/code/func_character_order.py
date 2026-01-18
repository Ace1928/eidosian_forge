from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def character_order(self) -> str:
    """The text direction for the language.

        >>> Locale('de', 'DE').character_order
        'left-to-right'
        >>> Locale('ar', 'SA').character_order
        'right-to-left'
        """
    return self._data['character_order']