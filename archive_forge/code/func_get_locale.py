import sys
from math import trunc
from typing import (
def get_locale(name: str) -> 'Locale':
    """Returns an appropriate :class:`Locale <arrow.locales.Locale>`
    corresponding to an input locale name.

    :param name: the name of the locale.

    """
    normalized_locale_name = name.lower().replace('_', '-')
    locale_cls = _locale_map.get(normalized_locale_name)
    if locale_cls is None:
        raise ValueError(f'Unsupported locale {normalized_locale_name!r}.')
    return locale_cls()