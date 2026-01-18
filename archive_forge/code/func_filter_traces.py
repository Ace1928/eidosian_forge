from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
def filter_traces(self, filters):
    """
        Create a new Snapshot instance with a filtered traces sequence, filters
        is a list of Filter or DomainFilter instances.  If filters is an empty
        list, return a new Snapshot instance with a copy of the traces.
        """
    if not isinstance(filters, Iterable):
        raise TypeError('filters must be a list of filters, not %s' % type(filters).__name__)
    if filters:
        include_filters = []
        exclude_filters = []
        for trace_filter in filters:
            if trace_filter.inclusive:
                include_filters.append(trace_filter)
            else:
                exclude_filters.append(trace_filter)
        new_traces = [trace for trace in self.traces._traces if self._filter_trace(include_filters, exclude_filters, trace)]
    else:
        new_traces = self.traces._traces.copy()
    return Snapshot(new_traces, self.traceback_limit)