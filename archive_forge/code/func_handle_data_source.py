from __future__ import annotations
from collections.abc import Mapping, Sized
from typing import cast
import warnings
import pandas as pd
from pandas import DataFrame
from seaborn._core.typing import DataSource, VariableSpec, ColumnName
from seaborn.utils import _version_predates
def handle_data_source(data: object) -> pd.DataFrame | Mapping | None:
    """Convert the data source object to a common union representation."""
    if isinstance(data, pd.DataFrame) or hasattr(data, '__dataframe__'):
        data = convert_dataframe_to_pandas(data)
    elif data is not None and (not isinstance(data, Mapping)):
        err = f'Data source must be a DataFrame or Mapping, not {type(data)!r}.'
        raise TypeError(err)
    return data