import locale
import threading
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, Union
from urllib.parse import urljoin, urlsplit
from .exceptions import xpath_error
def get_locale_category(category: int) -> str:
    """
    Gets the current value of a locale category. A replacement
    of locale.getdefaultlocale(), deprecated since Python 3.11.
    """
    _locale = locale.setlocale(category, None)
    if _locale == 'C':
        _locale = locale.setlocale(category, '')
        locale.setlocale(category, 'C')
    return _locale