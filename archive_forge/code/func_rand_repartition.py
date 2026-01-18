from typing import Any, Iterable, List
import pyspark.sql as ps
import pyspark.sql.functions as psf
from pyspark import RDD
from pyspark.sql import SparkSession
import warnings
from .convert import to_schema, to_spark_schema
from .misc import is_spark_connect
def rand_repartition(session: SparkSession, df: ps.DataFrame, num: int, cols: List[Any]) -> ps.DataFrame:
    if len(cols) > 0 or num <= 1:
        return hash_repartition(session, df, num, cols)
    tdf = df.withColumn(_PARTITION_DUMMY_KEY, (psf.rand(0) * psf.lit(2 ** 15 - 1)).cast('long'))
    return tdf.repartition(num, _PARTITION_DUMMY_KEY)[df.schema.names]