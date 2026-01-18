import sys
from math import trunc
from typing import (
def get_locale_by_class_name(name: str) -> 'Locale':
    """Returns an appropriate :class:`Locale <arrow.locales.Locale>`
    corresponding to an locale class name.

    :param name: the name of the locale class.

    """
    locale_cls: Optional[Type[Locale]] = globals().get(name)
    if locale_cls is None:
        raise ValueError(f'Unsupported locale {name!r}.')
    return locale_cls()