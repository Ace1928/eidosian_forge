import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
def list_records(self):
    """Get the processed data for the timing report.

        Returns
        -------
        res: List[PassTimingRecord]
        """
    return self._processed