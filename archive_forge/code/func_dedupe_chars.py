import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def dedupe_chars(chars: list, tolerance=1) -> list:
    """
    Removes duplicate chars —\xa0those sharing the same text, fontname, size,
    and positioning (within `tolerance`) as other characters in the set.
    """
    key = itemgetter('fontname', 'size', 'upright', 'text')
    pos_key = itemgetter('doctop', 'x0')

    def yield_unique_chars(chars: list):
        sorted_chars = sorted(chars, key=key)
        for grp, grp_chars in itertools.groupby(sorted_chars, key=key):
            for y_cluster in cluster_objects(list(grp_chars), itemgetter('doctop'), tolerance):
                for x_cluster in cluster_objects(y_cluster, itemgetter('x0'), tolerance):
                    yield sorted(x_cluster, key=pos_key)[0]
    deduped = yield_unique_chars(chars)
    return sorted(deduped, key=chars.index)