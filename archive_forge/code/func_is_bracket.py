import re
import unicodedata
from typing import Set
from .. import attrs
from .tokenizer_exceptions import URL_MATCH
def is_bracket(text: str) -> bool:
    brackets = ('(', ')', '[', ']', '{', '}', '<', '>')
    return text in brackets