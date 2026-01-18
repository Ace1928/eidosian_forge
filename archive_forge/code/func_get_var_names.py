from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
from pandas.util._decorators import Appender
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.indexes.api import MultiIndex
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import tile_compat
from pandas.core.shared_docs import _shared_docs
from pandas.core.tools.numeric import to_numeric
def get_var_names(df, stub: str, sep: str, suffix: str):
    regex = f'^{re.escape(stub)}{re.escape(sep)}{suffix}$'
    return df.columns[df.columns.str.match(regex)]