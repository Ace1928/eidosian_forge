from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
def compare_to(self, old_snapshot, key_type, cumulative=False):
    """
        Compute the differences with an old snapshot old_snapshot. Get
        statistics as a sorted list of StatisticDiff instances, grouped by
        group_by.
        """
    new_group = self._group_by(key_type, cumulative)
    old_group = old_snapshot._group_by(key_type, cumulative)
    statistics = _compare_grouped_stats(old_group, new_group)
    statistics.sort(reverse=True, key=StatisticDiff._sort_key)
    return statistics