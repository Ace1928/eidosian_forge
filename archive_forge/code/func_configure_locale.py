from __future__ import annotations
import locale
import sys
import typing as t
def configure_locale() -> tuple[str, t.Optional[str]]:
    """Configure the locale, returning the selected locale and an optional warning."""
    if (fs_encoding := sys.getfilesystemencoding()).lower() != 'utf-8':
        raise LocaleError(f'ansible-test requires the filesystem encoding to be UTF-8, but "{fs_encoding}" was detected.')
    candidate_locales = (STANDARD_LOCALE, FALLBACK_LOCALE)
    errors: dict[str, str] = {}
    warning: t.Optional[str] = None
    configured_locale: t.Optional[str] = None
    for candidate_locale in candidate_locales:
        try:
            locale.setlocale(locale.LC_ALL, candidate_locale)
            locale.getlocale()
        except (locale.Error, ValueError) as ex:
            errors[candidate_locale] = str(ex)
        else:
            configured_locale = candidate_locale
            break
    if not configured_locale:
        raise LocaleError('ansible-test could not initialize a supported locale:\n' + '\n'.join((f'{key}: {value}' for key, value in errors.items())))
    if configured_locale != STANDARD_LOCALE:
        warning = f'Using locale "{configured_locale}" instead of "{STANDARD_LOCALE}". Tests which depend on the locale may behave unexpectedly.'
    return (configured_locale, warning)