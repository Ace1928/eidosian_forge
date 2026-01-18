from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
def get_column_by_name(self, name: str) -> PandasColumn:
    return PandasColumn(self._df[name], allow_copy=self._allow_copy)