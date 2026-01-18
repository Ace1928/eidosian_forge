import pickle
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import ray
import ray.data as rd
from triad import Schema
from .._constants import _ZERO_COPY
def add_partition_key(df: rd.Dataset, input_schema: Schema, keys: List[str], output_key: str) -> Tuple[rd.Dataset, Schema]:

    def is_valid_type(tp: pa.DataType) -> bool:
        return pa.types.is_string(tp) or pa.types.is_integer(tp) or pa.types.is_floating(tp) or pa.types.is_date(tp) or pa.types.is_time(tp) or pa.types.is_timestamp(tp) or pa.types.is_boolean(tp) or pa.types.is_binary(tp)
    ray_remote_args: Dict[str, Any] = {'num_cpus': 1}
    if len(keys) == 1 and is_valid_type(input_schema[keys[0]].type):

        def add_simple_key(arrow_df: pa.Table) -> pa.Table:
            return arrow_df.append_column(output_key, arrow_df.column(input_schema.index_of_key(keys[0])).cast(pa.string()).fill_null(_RAY_NULL_REPR))
        return (df.map_batches(add_simple_key, batch_format='pyarrow', **_ZERO_COPY, **ray_remote_args), input_schema + (output_key, str))
    else:
        key_cols = [input_schema.index_of_key(k) for k in keys]

        def add_key(arrow_df: pa.Table) -> pa.Table:
            fdf = arrow_df.combine_chunks()
            sarr = pa.StructArray.from_arrays([fdf.column(i).combine_chunks() for i in key_cols], keys).tolist()
            sarr = pa.array([pickle.dumps(x) for x in sarr])
            return fdf.append_column(output_key, sarr)
        return (df.map_batches(add_key, batch_format='pyarrow', **_ZERO_COPY, **ray_remote_args), input_schema + (output_key, pa.binary()))