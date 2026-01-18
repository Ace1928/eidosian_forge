import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def is_atomic(self):
    return all((s.is_atomic() for s in self.items))