from typing import Any, Iterable, List
import pyspark.sql as ps
import pyspark.sql.functions as psf
from pyspark import RDD
from pyspark.sql import SparkSession
import warnings
from .convert import to_schema, to_spark_schema
from .misc import is_spark_connect
def hash_repartition(session: SparkSession, df: ps.DataFrame, num: int, cols: List[Any]) -> ps.DataFrame:
    if num <= 0:
        if len(cols) == 0:
            return df
        return df.repartition(*cols)
    if num == 1:
        return _single_repartition(df)
    return df.repartition(num, *cols)