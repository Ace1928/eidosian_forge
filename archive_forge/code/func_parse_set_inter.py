import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_set_inter(source, info):
    """Parses a set intersection ([x&&y])."""
    items = [parse_set_diff(source, info)]
    while source.match('&&'):
        items.append(parse_set_diff(source, info))
    if len(items) == 1:
        return items[0]
    return SetInter(info, items)