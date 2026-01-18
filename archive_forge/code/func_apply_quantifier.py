import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def apply_quantifier(source, info, counts, case_flags, ch, saved_pos, sequence):
    element = sequence.pop()
    if element is None:
        if sequence:
            raise error('multiple repeat', source.string, saved_pos)
        raise error('nothing to repeat', source.string, saved_pos)
    if isinstance(element, (GreedyRepeat, LazyRepeat, PossessiveRepeat)):
        raise error('multiple repeat', source.string, saved_pos)
    min_count, max_count = counts
    saved_pos = source.pos
    ch = source.get()
    if ch == '?':
        repeated = LazyRepeat
    elif ch == '+':
        repeated = PossessiveRepeat
    else:
        source.pos = saved_pos
        repeated = GreedyRepeat
    if not element.is_empty() and (min_count != 1 or max_count != 1):
        element = repeated(element, min_count, max_count)
    sequence.append(element)