from __future__ import annotations
from collections.abc import (
import functools
import itertools
import re
from typing import (
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes import missing
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.shared_docs import _shared_docs
from pandas.io.formats._color_data import CSS4_COLORS
from pandas.io.formats.css import (
from pandas.io.formats.format import get_level_lengths
from pandas.io.formats.printing import pprint_thing
def color_to_excel(self, val: str | None) -> str | None:
    if val is None:
        return None
    if self._is_hex_color(val):
        return self._convert_hex_to_excel(val)
    try:
        return self.NAMED_COLORS[val]
    except KeyError:
        warnings.warn(f'Unhandled color format: {repr(val)}', CSSWarning, stacklevel=find_stack_level())
    return None