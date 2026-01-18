from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
def get_locale_identifier(tup: tuple[str] | tuple[str, str | None] | tuple[str, str | None, str | None] | tuple[str, str | None, str | None, str | None] | tuple[str, str | None, str | None, str | None, str | None], sep: str='_') -> str:
    """The reverse of :func:`parse_locale`.  It creates a locale identifier out
    of a ``(language, territory, script, variant, modifier)`` tuple.  Items can be set to
    ``None`` and trailing ``None``\\s can also be left out of the tuple.

    >>> get_locale_identifier(('de', 'DE', None, '1999', 'custom'))
    'de_DE_1999@custom'
    >>> get_locale_identifier(('fi', None, None, None, 'custom'))
    'fi@custom'


    .. versionadded:: 1.0

    :param tup: the tuple as returned by :func:`parse_locale`.
    :param sep: the separator for the identifier.
    """
    tup = tuple(tup[:5])
    lang, territory, script, variant, modifier = tup + (None,) * (5 - len(tup))
    ret = sep.join(filter(None, (lang, script, territory, variant)))
    return f'{ret}@{modifier}' if modifier else ret