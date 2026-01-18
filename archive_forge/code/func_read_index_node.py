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
def read_index_node(self, node: Node, start: int | None=None, stop: int | None=None) -> Index:
    data = node[start:stop]
    if 'shape' in node._v_attrs and np.prod(node._v_attrs.shape) == 0:
        data = np.empty(node._v_attrs.shape, dtype=node._v_attrs.value_type)
    kind = _ensure_decoded(node._v_attrs.kind)
    name = None
    if 'name' in node._v_attrs:
        name = _ensure_str(node._v_attrs.name)
        name = _ensure_decoded(name)
    attrs = node._v_attrs
    factory, kwargs = self._get_index_factory(attrs)
    if kind in ('date', 'object'):
        index = factory(_unconvert_index(data, kind, encoding=self.encoding, errors=self.errors), dtype=object, **kwargs)
    else:
        index = factory(_unconvert_index(data, kind, encoding=self.encoding, errors=self.errors), **kwargs)
    index.name = name
    return index