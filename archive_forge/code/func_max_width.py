import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def max_width(self):
    return len(self.folded_characters)