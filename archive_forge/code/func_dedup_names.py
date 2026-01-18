from __future__ import annotations
from abc import (
import codecs
from collections import defaultdict
from collections.abc import (
import dataclasses
import functools
import gzip
from io import (
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import (
from urllib.parse import (
import warnings
import zipfile
from pandas._typing import (
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCMultiIndex
from pandas.core.shared_docs import _shared_docs
def dedup_names(names: Sequence[Hashable], is_potential_multiindex: bool) -> Sequence[Hashable]:
    """
    Rename column names if duplicates exist.

    Currently the renaming is done by appending a period and an autonumeric,
    but a custom pattern may be supported in the future.

    Examples
    --------
    >>> dedup_names(["x", "y", "x", "x"], is_potential_multiindex=False)
    ['x', 'y', 'x.1', 'x.2']
    """
    names = list(names)
    counts: DefaultDict[Hashable, int] = defaultdict(int)
    for i, col in enumerate(names):
        cur_count = counts[col]
        while cur_count > 0:
            counts[col] = cur_count + 1
            if is_potential_multiindex:
                assert isinstance(col, tuple)
                col = col[:-1] + (f'{col[-1]}.{cur_count}',)
            else:
                col = f'{col}.{cur_count}'
            cur_count = counts[col]
        names[i] = col
        counts[col] = cur_count + 1
    return names