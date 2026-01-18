from typing import Any, Callable, Dict, List, Tuple
import dask.array as np
import dask.dataframe as pd
import numpy
import pandas
from triad.utils.assertion import assert_or_throw
from triad.utils.pandas_like import PandasLikeUtils
from qpd import QPDEngine, run_sql
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_pandas.engine import _RowsIndexer
def run_sql_on_dask(sql: str, dfs: Dict[str, Any], ignore_case: bool=False) -> pd.DataFrame:
    return run_sql(QPDDaskEngine(), sql, dfs, ignore_case=ignore_case)