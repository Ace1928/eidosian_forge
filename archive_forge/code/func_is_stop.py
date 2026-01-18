import re
import unicodedata
from typing import Set
from .. import attrs
from .tokenizer_exceptions import URL_MATCH
def is_stop(string: str, stops: Set[str]=set()) -> bool:
    return string.lower() in stops