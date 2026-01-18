from __future__ import annotations
from collections import deque
import collections.abc as collections_abc
import itertools
from itertools import zip_longest
import operator
import typing
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from . import operators
from .cache_key import HasCacheKey
from .visitors import _TraverseInternalsType
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .. import util
from ..util import langhelpers
from ..util.typing import Self
def compare_binary(self, left, right, **kw):
    if left.operator == right.operator:
        if operators.is_commutative(left.operator):
            if self.compare_inner(left.left, right.left, **kw) and self.compare_inner(left.right, right.right, **kw) or (self.compare_inner(left.left, right.right, **kw) and self.compare_inner(left.right, right.left, **kw)):
                return ['operator', 'negate', 'left', 'right']
            else:
                return COMPARE_FAILED
        else:
            return ['operator', 'negate']
    else:
        return COMPARE_FAILED