from __future__ import annotations
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import TypeVar
from ..orm.collections import collection
from ..orm.collections import collection_adapter
def count_from_1(index, collection):
    """Numbering function: consecutive integers starting at 1."""
    return index + 1