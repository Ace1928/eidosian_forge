import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
from fsspec import AbstractFileSystem
from triad import Schema, assert_or_throw
from triad.collections.schema import SchemaError
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.io import url_to_fs
from triad.utils.pyarrow import pa_batch_to_dicts
from .api import as_fugue_df, get_column_names, normalize_column_names, rename
from .dataframe import DataFrame, LocalBoundedDataFrame
def get_join_schemas(df1: DataFrame, df2: DataFrame, how: str, on: Optional[Iterable[str]]) -> Tuple[Schema, Schema]:
    """Get :class:`~triad:triad.collections.schema.Schema` object after
    joining ``df1`` and ``df2``. If ``on`` is not empty, it's mainly for
    validation purpose.

    :param df1: first dataframe
    :param df2: second dataframe
    :param how: can accept ``semi``, ``left_semi``, ``anti``, ``left_anti``,
      ``inner``, ``left_outer``, ``right_outer``, ``full_outer``, ``cross``
    :param on: it can always be inferred, but if you provide, it will be
      validated agained the inferred keys.
    :return: the pair key schema and schema after join

    .. note::

        In Fugue, joined schema can always be inferred because it always uses the
        input dataframes' common keys as the join keys. So you must make sure to
        :meth:`~fugue.dataframe.dataframe.DataFrame.rename` to input dataframes so
        they follow this rule.
    """
    assert_arg_not_none(how, 'how')
    how = how.lower()
    aot(how in ['semi', 'left_semi', 'anti', 'left_anti', 'inner', 'left_outer', 'right_outer', 'full_outer', 'cross'], ValueError(f'{how} is not a valid join type'))
    on = list(on) if on is not None else []
    aot(len(on) == len(set(on)), f'{on} has duplication')
    if how != 'cross' and len(on) == 0:
        other = set(df2.columns)
        on = [c for c in df1.columns if c in other]
        aot(len(on) > 0, lambda: SchemaError(f'no common columns between {df1.columns} and {df2.columns}'))
    schema2 = df2.schema
    aot(how != 'outer', ValueError("'how' must use left_outer, right_outer, full_outer for outer joins"))
    if how in ['semi', 'left_semi', 'anti', 'left_anti']:
        schema2 = schema2.extract(on)
    aot(on in df1.schema and on in schema2, lambda: SchemaError(f'{on} is not the intersection of {df1.schema} & {df2.schema}'))
    cm = df1.schema.intersect(on)
    if how == 'cross':
        cs = df1.schema.intersect(schema2)
        aot(len(cs) == 0, SchemaError(f'invalid cross join, two dataframes have common columns {cs}'))
    else:
        aot(len(on) > 0, SchemaError('join on columns must be specified'))
    return (cm, df1.schema.union(schema2))