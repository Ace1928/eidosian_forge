import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def iter_chars_to_words(self, ordered_chars):
    current_word: list = []

    def start_next_word(new_char=None):
        nonlocal current_word
        if current_word:
            yield current_word
        current_word = [] if new_char is None else [new_char]
    for char in ordered_chars:
        text = char['text']
        if not self.keep_blank_chars and text.isspace():
            yield from start_next_word(None)
        elif text in self.split_at_punctuation:
            yield from start_next_word(char)
            yield from start_next_word(None)
        elif current_word and self.char_begins_new_word(current_word[-1], char):
            yield from start_next_word(char)
        else:
            current_word.append(char)
    if current_word:
        yield current_word