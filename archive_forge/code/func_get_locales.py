from __future__ import annotations
from contextlib import contextmanager
import locale
import platform
import re
import subprocess
from typing import TYPE_CHECKING
from pandas._config.config import options
def get_locales(prefix: str | None=None, normalize: bool=True) -> list[str]:
    """
    Get all the locales that are available on the system.

    Parameters
    ----------
    prefix : str
        If not ``None`` then return only those locales with the prefix
        provided. For example to get all English language locales (those that
        start with ``"en"``), pass ``prefix="en"``.
    normalize : bool
        Call ``locale.normalize`` on the resulting list of available locales.
        If ``True``, only locales that can be set without throwing an
        ``Exception`` are returned.

    Returns
    -------
    locales : list of strings
        A list of locale strings that can be set with ``locale.setlocale()``.
        For example::

            locale.setlocale(locale.LC_ALL, locale_string)

    On error will return an empty list (no locale available, e.g. Windows)

    """
    if platform.system() in ('Linux', 'Darwin'):
        raw_locales = subprocess.check_output(['locale', '-a'])
    else:
        return []
    try:
        split_raw_locales = raw_locales.split(b'\n')
        out_locales = []
        for x in split_raw_locales:
            try:
                out_locales.append(str(x, encoding=options.display.encoding))
            except UnicodeError:
                out_locales.append(str(x, encoding='windows-1252'))
    except TypeError:
        pass
    if prefix is None:
        return _valid_locales(out_locales, normalize)
    pattern = re.compile(f'{prefix}.*')
    found = pattern.findall('\n'.join(out_locales))
    return _valid_locales(found, normalize)