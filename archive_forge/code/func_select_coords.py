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
def select_coords(self):
    """
        generate the selection
        """
    start, stop = (self.start, self.stop)
    nrows = self.table.nrows
    if start is None:
        start = 0
    elif start < 0:
        start += nrows
    if stop is None:
        stop = nrows
    elif stop < 0:
        stop += nrows
    if self.condition is not None:
        return self.table.table.get_where_list(self.condition.format(), start=start, stop=stop, sort=True)
    elif self.coordinates is not None:
        return self.coordinates
    return np.arange(start, stop)