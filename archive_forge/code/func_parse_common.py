import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def parse_common(source, info):
    """Parses a common groups branch."""
    initial_group_count = info.group_count
    branches = [parse_sequence(source, info)]
    final_group_count = info.group_count
    while source.match('|'):
        info.group_count = initial_group_count
        branches.append(parse_sequence(source, info))
        final_group_count = max(final_group_count, info.group_count)
    info.group_count = final_group_count
    source.expect(')')
    if len(branches) == 1:
        return branches[0]
    return Branch(branches)