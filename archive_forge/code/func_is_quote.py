import re
import unicodedata
from typing import Set
from .. import attrs
from .tokenizer_exceptions import URL_MATCH
def is_quote(text: str) -> bool:
    quotes = ('"', "'", '`', '«', '»', '‘', '’', '‚', '‛', '“', '”', '„', '‟', '‹', '›', '❮', '❯', "''", '``')
    return text in quotes