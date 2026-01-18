from __future__ import annotations
from typing import Callable
from pandas.compat._optional import import_optional_dependency
import pandas as pd
def arrow_string_types_mapper() -> Callable:
    pa = import_optional_dependency('pyarrow')
    return {pa.string(): pd.StringDtype(storage='pyarrow_numpy'), pa.large_string(): pd.StringDtype(storage='pyarrow_numpy')}.get