from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def detect_colspecs(self, infer_nrows: int=100, skiprows: set[int] | None=None) -> list[tuple[int, int]]:
    delimiters = ''.join([f'\\{x}' for x in self.delimiter])
    pattern = re.compile(f'([^{delimiters}]+)')
    rows = self.get_rows(infer_nrows, skiprows)
    if not rows:
        raise EmptyDataError('No rows from which to infer column width')
    max_len = max(map(len, rows))
    mask = np.zeros(max_len + 1, dtype=int)
    if self.comment is not None:
        rows = [row.partition(self.comment)[0] for row in rows]
    for row in rows:
        for m in pattern.finditer(row):
            mask[m.start():m.end()] = 1
    shifted = np.roll(mask, 1)
    shifted[0] = 0
    edges = np.where(mask ^ shifted == 1)[0]
    edge_pairs = list(zip(edges[::2], edges[1::2]))
    return edge_pairs