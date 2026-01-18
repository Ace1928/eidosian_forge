import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def is_open_group(self, name):
    version = self.flags & _ALL_VERSIONS or DEFAULT_VERSION
    if version == VERSION1:
        return False
    if name.isdigit():
        group = int(name)
    else:
        group = self.group_index.get(name)
    return group in self.open_groups