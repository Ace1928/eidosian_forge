import re
import unicodedata
from typing import Set
from .. import attrs
from .tokenizer_exceptions import URL_MATCH
def get_lang(text: str, lang: str='') -> str:
    return lang