from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
@cache_readonly
def indexables(self):
    """create the indexables from the table description"""
    d = self.description
    md = self.read_metadata('index')
    meta = 'category' if md is not None else None
    index_col = GenericIndexCol(name='index', axis=0, table=self.table, meta=meta, metadata=md)
    _indexables: list[GenericIndexCol | GenericDataIndexableCol] = [index_col]
    for i, n in enumerate(d._v_names):
        assert isinstance(n, str)
        atom = getattr(d, n)
        md = self.read_metadata(n)
        meta = 'category' if md is not None else None
        dc = GenericDataIndexableCol(name=n, pos=i, values=[n], typ=atom, table=self.table, meta=meta, metadata=md)
        _indexables.append(dc)
    return _indexables