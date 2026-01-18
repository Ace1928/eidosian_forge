from collections import defaultdict
from operator import itemgetter
from time import time
from typing import TYPE_CHECKING, Any, DefaultDict, Iterable
from weakref import WeakKeyDictionary
def print_live_refs(*a: Any, **kw: Any) -> None:
    """Print tracked objects"""
    print(format_live_refs(*a, **kw))