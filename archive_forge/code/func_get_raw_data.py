import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
def get_raw_data(self):
    """Returns the raw string data.

        Returns
        -------
        res: str
        """
    return self._raw_data