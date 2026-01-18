import pickle
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pyarrow as pa
import ray
import ray.data as rd
from triad import Schema
from .._constants import _ZERO_COPY
def get_dataset_format(df: rd.Dataset) -> Tuple[Optional[str], rd.Dataset]:
    df = materialize(df)
    if df.count() == 0:
        return (None, df)
    if ray.__version__ < '2.5.0':
        if hasattr(df, '_dataset_format'):
            return (df._dataset_format(), df)
        ctx = rd.context.DatasetContext.get_current()
        ctx.use_streaming_executor = False
        return (df.dataset_format(), df)
    else:
        schema = df.schema(fetch_if_missing=True)
        if schema is None:
            return (None, df)
        if isinstance(schema.base_schema, pa.Schema):
            return ('arrow', df)
        return ('pandas', df)