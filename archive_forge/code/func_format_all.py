from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path
@staticmethod
def format_all(tbs):
    """
        Bulk version of CapturedTraceback.format.  Returns a list of list of strings.
        """
    import torch._C._profiler
    rs: List[Optional[List[str]]] = []
    delayed_idxs = []
    for i, tb in enumerate(tbs):
        if tb.tb is None:
            rs.append([])
        else:
            rs.append(None)
            delayed_idxs.append(i)
    stbs = torch._C._profiler.symbolize_tracebacks([tbs[i].tb for i in delayed_idxs])
    for i, stb in zip(delayed_idxs, stbs):
        rs[i] = traceback.format_list(tbs[i].summary())
    return rs