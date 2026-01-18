import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
def list_top(self, n):
    """Returns the top(n) most time-consuming (by wall-time) passes.

        Parameters
        ----------
        n: int
            This limits the maximum number of items to show.
            This function will show the ``n`` most time-consuming passes.

        Returns
        -------
        res: List[PassTimingRecord]
            Returns the top(n) most time-consuming passes in descending order.
        """
    records = self.list_records()
    key = operator.attrgetter('wall_time')
    return heapq.nlargest(n, records[:-1], key)