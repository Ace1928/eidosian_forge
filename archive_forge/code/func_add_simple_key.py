import pickle
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import ray
import ray.data as rd
from triad import Schema
from .._constants import _ZERO_COPY
def add_simple_key(arrow_df: pa.Table) -> pa.Table:
    return arrow_df.append_column(output_key, arrow_df.column(input_schema.index_of_key(keys[0])).cast(pa.string()).fill_null(_RAY_NULL_REPR))