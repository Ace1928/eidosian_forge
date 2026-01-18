import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def iter_extract_tuples(self, chars):
    ordered_chars = chars if self.use_text_flow else self.iter_sort_chars(chars)
    grouping_key = itemgetter('upright', *self.extra_attrs)
    grouped_chars = itertools.groupby(ordered_chars, grouping_key)
    for keyvals, char_group in grouped_chars:
        for word_chars in self.iter_chars_to_words(char_group):
            yield (self.merge_chars(word_chars), word_chars)