from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
def get_language_name(self, locale: Locale | str | None=None) -> str | None:
    """Return the language of this locale in the given locale.

        >>> Locale('zh', 'CN', script='Hans').get_language_name('de')
        u'Chinesisch'

        .. versionadded:: 1.0

        :param locale: the locale to use
        """
    if locale is None:
        locale = self
    locale = Locale.parse(locale)
    return locale.languages.get(self.language)