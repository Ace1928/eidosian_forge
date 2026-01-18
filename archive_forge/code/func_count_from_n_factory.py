from __future__ import annotations
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar
from ..orm.collections import collection
from ..orm.collections import collection_adapter
def count_from_n_factory(start):
    """Numbering function: consecutive integers starting at arbitrary start."""

    def f(index, collection):
        return index + start
    try:
        f.__name__ = 'count_from_%i' % start
    except TypeError:
        pass
    return f