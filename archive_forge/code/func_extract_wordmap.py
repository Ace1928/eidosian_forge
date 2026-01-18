import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def extract_wordmap(self, chars) -> WordMap:
    return WordMap(list(self.iter_extract_tuples(chars)))