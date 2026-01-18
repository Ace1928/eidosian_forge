import logging
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union
import numpy as np
import pandas as pd
from triad import Schema
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_or_throw
from triad.utils.io import makedirs
from triad.utils.pandas_like import PandasUtils
from fugue._utils.io import load_df, save_df
from fugue._utils.misc import import_fsql_dependency
from fugue.collections.partition import (
from fugue.collections.sql import StructuredRawSQL
from fugue.constants import KEYWORD_PARALLELISM, KEYWORD_ROWCOUNT
from fugue.dataframe import (
from fugue.dataframe.dataframe import as_fugue_df
from fugue.dataframe.utils import get_join_schemas
from .execution_engine import (
def map_dataframe(self, df: DataFrame, map_func: Callable[[PartitionCursor, LocalDataFrame], LocalDataFrame], output_schema: Any, partition_spec: PartitionSpec, on_init: Optional[Callable[[int, DataFrame], Any]]=None, map_func_format_hint: Optional[str]=None) -> DataFrame:
    is_coarse = partition_spec.algo == 'coarse'
    presort = partition_spec.get_sorts(df.schema, with_partition_keys=is_coarse)
    presort_keys = list(presort.keys())
    presort_asc = list(presort.values())
    output_schema = Schema(output_schema)
    cursor = partition_spec.get_cursor(df.schema, 0)
    if on_init is not None:
        on_init(0, df)
    if len(partition_spec.partition_by) == 0 or partition_spec.algo == 'coarse':
        if len(partition_spec.presort) > 0:
            pdf = df.as_pandas().sort_values(presort_keys, ascending=presort_asc).reset_index(drop=True)
            input_df: LocalDataFrame = PandasDataFrame(pdf, df.schema, pandas_df_wrapper=True)
        else:
            input_df = df.as_local()
        if len(partition_spec.partition_by) == 0 and partition_spec.num_partitions != '0':
            partitions = partition_spec.get_num_partitions(**{KEYWORD_ROWCOUNT: lambda: df.count(), KEYWORD_PARALLELISM: lambda: 1})
            dfs: List[DataFrame] = []
            for p, subdf in enumerate(np.array_split(input_df.as_pandas(), partitions)):
                if len(subdf) > 0:
                    tdf = PandasDataFrame(subdf, df.schema, pandas_df_wrapper=True)
                    cursor.set(lambda: tdf.peek_array(), p, 0)
                    dfs.append(map_func(cursor, tdf).as_pandas())
            output_df: LocalDataFrame = PandasDataFrame(pd.concat(dfs, ignore_index=True), schema=output_schema, pandas_df_wrapper=True)
        else:
            cursor.set(lambda: input_df.peek_array(), 0, 0)
            output_df = map_func(cursor, input_df)
        if isinstance(output_df, PandasDataFrame) and output_df.schema != output_schema:
            output_df = PandasDataFrame(output_df.native, output_schema)
        assert_or_throw(output_df.schema == output_schema, lambda: f'map output {output_df.schema} mismatches given {output_schema}')
        return self.to_df(output_df)

    def _map(pdf: pd.DataFrame) -> pd.DataFrame:
        if len(partition_spec.presort) > 0:
            pdf = pdf.sort_values(presort_keys, ascending=presort_asc).reset_index(drop=True)
        input_df = PandasDataFrame(pdf, df.schema, pandas_df_wrapper=True)
        cursor.set(lambda: input_df.peek_array(), cursor.partition_no + 1, 0)
        output_df = map_func(cursor, input_df)
        return output_df.as_pandas()
    result = self.execution_engine.pl_utils.safe_groupby_apply(df.as_pandas(), partition_spec.partition_by, _map)
    return PandasDataFrame(result, output_schema)