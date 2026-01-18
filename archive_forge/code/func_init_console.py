import locale
from gettext import NullTranslations, translation
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
def init_console(locale_dir: str=_LOCALE_DIR, catalog: str='sphinx') -> Tuple[NullTranslations, bool]:
    """Initialize locale for console.

    .. versionadded:: 1.8
    """
    try:
        language, _ = locale.getlocale(locale.LC_MESSAGES)
    except AttributeError:
        language = None
    return init([locale_dir], language, catalog, 'console')