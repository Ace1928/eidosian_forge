import re
import unicodedata
from typing import Set
from .. import attrs
from .tokenizer_exceptions import URL_MATCH
def is_right_punct(text: str) -> bool:
    right_punct = (')', ']', '}', '>', '"', "'", '»', '’', '”', '›', '❯', "''")
    return text in right_punct