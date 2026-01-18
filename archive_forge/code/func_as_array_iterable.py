from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def as_array_iterable(self, df: T, schema: Optional[pa.Schema]=None, columns: Optional[List[str]]=None, type_safe: bool=False) -> Iterable[List[Any]]:
    """Convert pandas like dataframe to iterable of rows in the format of list.

        :param df: pandas like dataframe
        :param schema: schema of the input. With None, it will infer the schema,
          it can't infer wrong schema for nested types, so try to be explicit
        :param columns: columns to output, None for all columns
        :param type_safe: whether to enforce the types in schema, if False, it will
            return the original values from the dataframe
        :return: iterable of rows, each row is a list
        """
    if self.empty(df):
        return
    if schema is None:
        schema = self.to_schema(df)
    if columns is not None:
        df = df[columns]
        schema = pa.schema([schema.field(n) for n in columns])
    if not type_safe:
        for arr in df.itertuples(index=False, name=None):
            yield list(arr)
    else:
        p = self.as_arrow(df, schema)
        d = p.to_pydict()
        cols = [d[n] for n in schema.names]
        for arr in zip(*cols):
            yield list(arr)