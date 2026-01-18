import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
def dfs_iter(self, include_self=False, right_to_left=True):
    """Depth first iteration (non-recursive) over the child nodes."""
    return _DFSIter(self, include_self=include_self, right_to_left=right_to_left)