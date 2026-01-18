import locale
from gettext import NullTranslations, translation
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
def get_translator(catalog: str='sphinx', namespace: str='general') -> NullTranslations:
    return translators.get((namespace, catalog), NullTranslations())