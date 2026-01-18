import locale
from gettext import NullTranslations, translation
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
def is_translator_registered(catalog: str='sphinx', namespace: str='general') -> bool:
    return (namespace, catalog) in translators