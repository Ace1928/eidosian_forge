from typing import Any, Callable, Dict, List, Optional, Union
import pyspark.sql as ps
from pyspark.sql import SparkSession
from triad.collections import Schema
from triad.collections.dict import ParamDict
from triad.utils.assertion import assert_or_throw
from fugue._utils.io import FileParser, save_df
from fugue.collections.partition import PartitionSpec
from fugue.dataframe import DataFrame, PandasDataFrame
from fugue_spark.dataframe import SparkDataFrame
from .convert import to_schema, to_spark_schema
def save_df(self, df: SparkDataFrame, uri: str, format_hint: Optional[str]=None, partition_spec: Optional[PartitionSpec]=None, mode: str='overwrite', force_single: bool=False, **kwargs: Any) -> None:
    partition_spec = partition_spec or PartitionSpec()
    if not force_single:
        p = FileParser(uri, format_hint)
        writer = self._get_writer(df.native, partition_spec)
        writer.format(p.file_format).options(**kwargs).mode(mode)
        writer.save(uri)
    else:
        ldf = df.as_local()
        if isinstance(ldf, PandasDataFrame) and hasattr(ldf.native, 'attrs'):
            ldf.native.attrs = {}
        save_df(ldf, uri, format_hint=format_hint, mode=mode, **kwargs)