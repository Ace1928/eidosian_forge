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
def build_xlstyle(self, props: Mapping[str, str]) -> dict[str, dict[str, str]]:
    out = {'alignment': self.build_alignment(props), 'border': self.build_border(props), 'fill': self.build_fill(props), 'font': self.build_font(props), 'number_format': self.build_number_format(props)}

    def remove_none(d: dict[str, str | None]) -> None:
        """Remove key where value is None, through nested dicts"""
        for k, v in list(d.items()):
            if v is None:
                del d[k]
            elif isinstance(v, dict):
                remove_none(v)
                if not v:
                    del d[k]
    remove_none(out)
    return out