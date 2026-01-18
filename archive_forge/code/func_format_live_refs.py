from collections import defaultdict
from operator import itemgetter
from time import time
from typing import TYPE_CHECKING, Any, DefaultDict, Iterable
from weakref import WeakKeyDictionary
def format_live_refs(ignore: Any=NoneType) -> str:
    """Return a tabular representation of tracked objects"""
    s = 'Live References\n\n'
    now = time()
    for cls, wdict in sorted(live_refs.items(), key=lambda x: x[0].__name__):
        if not wdict:
            continue
        if issubclass(cls, ignore):
            continue
        oldest = min(wdict.values())
        s += f'{cls.__name__:<30} {len(wdict):6}   oldest: {int(now - oldest)}s ago\n'
    return s