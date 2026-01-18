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
def read_multi_index(self, key: str, start: int | None=None, stop: int | None=None) -> MultiIndex:
    nlevels = getattr(self.attrs, f'{key}_nlevels')
    levels = []
    codes = []
    names: list[Hashable] = []
    for i in range(nlevels):
        level_key = f'{key}_level{i}'
        node = getattr(self.group, level_key)
        lev = self.read_index_node(node, start=start, stop=stop)
        levels.append(lev)
        names.append(lev.name)
        label_key = f'{key}_label{i}'
        level_codes = self.read_array(label_key, start=start, stop=stop)
        codes.append(level_codes)
    return MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=True)