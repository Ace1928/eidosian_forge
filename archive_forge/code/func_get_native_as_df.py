from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.utils.rename import normalize_names
from .._utils.registry import fugue_plugin
from .dataframe import AnyDataFrame, DataFrame, as_fugue_df
def get_native_as_df(df: AnyDataFrame) -> AnyDataFrame:
    """Return the dataframe form of any dataframe.
    If ``df`` is a :class:`~.DataFrame`, then call the
    :meth:`~.DataFrame.native_as_df`, otherwise, it depends on whether there is
    a correspondent function handling it.
    """
    if isinstance(df, DataFrame):
        return df.native_as_df()
    if is_df(df):
        return df
    raise NotImplementedError(f'cannot get a dataframe like object from {type(df)}')