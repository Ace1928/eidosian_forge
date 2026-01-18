from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from triad.collections.schema import Schema
from triad.utils.rename import normalize_names
from .._utils.registry import fugue_plugin
from .dataframe import AnyDataFrame, DataFrame, as_fugue_df
@fugue_plugin
def get_column_names(df: AnyDataFrame) -> List[Any]:
    """A generic function to get column names of any dataframe

    :param df: the dataframe object
    :return: the column names

    .. note::

        In order to support a new type of dataframe, an implementation must
        be registered, for example

        .. code-block::python

            @get_column_names.candidate(lambda df: isinstance(df, pa.Table))
            def _get_pyarrow_dataframe_columns(df: pa.Table) -> List[Any]:
                return [f.name for f in df.schema]
    """
    return get_schema(df).names