from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
def parse_locale(identifier: str, sep: str='_') -> tuple[str, str | None, str | None, str | None] | tuple[str, str | None, str | None, str | None, str | None]:
    """Parse a locale identifier into a tuple of the form ``(language,
    territory, script, variant, modifier)``.

    >>> parse_locale('zh_CN')
    ('zh', 'CN', None, None)
    >>> parse_locale('zh_Hans_CN')
    ('zh', 'CN', 'Hans', None)
    >>> parse_locale('ca_es_valencia')
    ('ca', 'ES', None, 'VALENCIA')
    >>> parse_locale('en_150')
    ('en', '150', None, None)
    >>> parse_locale('en_us_posix')
    ('en', 'US', None, 'POSIX')
    >>> parse_locale('it_IT@euro')
    ('it', 'IT', None, None, 'euro')
    >>> parse_locale('it_IT@custom')
    ('it', 'IT', None, None, 'custom')
    >>> parse_locale('it_IT@')
    ('it', 'IT', None, None)

    The default component separator is "_", but a different separator can be
    specified using the `sep` parameter.

    The optional modifier is always separated with "@" and at the end:

    >>> parse_locale('zh-CN', sep='-')
    ('zh', 'CN', None, None)
    >>> parse_locale('zh-CN@custom', sep='-')
    ('zh', 'CN', None, None, 'custom')

    If the identifier cannot be parsed into a locale, a `ValueError` exception
    is raised:

    >>> parse_locale('not_a_LOCALE_String')
    Traceback (most recent call last):
      ...
    ValueError: 'not_a_LOCALE_String' is not a valid locale identifier

    Encoding information is removed from the identifier, while modifiers are
    kept:

    >>> parse_locale('en_US.UTF-8')
    ('en', 'US', None, None)
    >>> parse_locale('de_DE.iso885915@euro')
    ('de', 'DE', None, None, 'euro')

    See :rfc:`4646` for more information.

    :param identifier: the locale identifier string
    :param sep: character that separates the different components of the locale
                identifier
    :raise `ValueError`: if the string does not appear to be a valid locale
                         identifier
    """
    identifier, _, modifier = identifier.partition('@')
    if '.' in identifier:
        identifier = identifier.split('.', 1)[0]
    parts = identifier.split(sep)
    lang = parts.pop(0).lower()
    if not lang.isalpha():
        raise ValueError(f'expected only letters, got {lang!r}')
    script = territory = variant = None
    if parts and len(parts[0]) == 4 and parts[0].isalpha():
        script = parts.pop(0).title()
    if parts:
        if len(parts[0]) == 2 and parts[0].isalpha():
            territory = parts.pop(0).upper()
        elif len(parts[0]) == 3 and parts[0].isdigit():
            territory = parts.pop(0)
    if parts and (len(parts[0]) == 4 and parts[0][0].isdigit() or (len(parts[0]) >= 5 and parts[0][0].isalpha())):
        variant = parts.pop().upper()
    if parts:
        raise ValueError(f'{identifier!r} is not a valid locale identifier')
    if modifier:
        return (lang, territory, script, variant, modifier)
    else:
        return (lang, territory, script, variant)