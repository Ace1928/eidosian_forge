from __future__ import annotations
import os
import sys
from typing import Any, TypeVar, Callable, Optional, NamedTuple
from typing_extensions import TypeAlias
from .._extras import pandas as pd
def lower_case_column(df: pd.DataFrame, column: Any) -> pd.DataFrame:
    cols = [c for c in df.columns if str(c).lower() == column]
    df.rename(columns={cols[0]: column.lower()}, inplace=True)
    return df