import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def chars_to_textmap(chars: list, **kwargs) -> TextMap:
    kwargs.update({'presorted': True})
    extractor = WordExtractor(**{k: kwargs[k] for k in WORD_EXTRACTOR_KWARGS if k in kwargs})
    wordmap = extractor.extract_wordmap(chars)
    textmap = wordmap.to_textmap(**{k: kwargs[k] for k in TEXTMAP_KWARGS if k in kwargs})
    return textmap