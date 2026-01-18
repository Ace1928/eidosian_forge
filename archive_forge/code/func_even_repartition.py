from typing import Any, Iterable, List
import pyspark.sql as ps
import pyspark.sql.functions as psf
from pyspark import RDD
from pyspark.sql import SparkSession
import warnings
from .convert import to_schema, to_spark_schema
from .misc import is_spark_connect
def even_repartition(session: SparkSession, df: ps.DataFrame, num: int, cols: List[Any]) -> ps.DataFrame:
    if num == 1:
        return _single_repartition(df)
    if is_spark_connect(session):
        warnings.warn('Even repartitioning is not supported by Spark Connect')
        return hash_repartition(session, df, num, cols)
    if len(cols) == 0:
        if num == 0:
            return df
        rdd = _zipWithIndex(df.rdd).partitionBy(num, lambda k: k).mapPartitions(_to_rows)
        return session.createDataFrame(rdd, df.schema)
    else:
        keys = df.select(*cols).distinct()
        krdd = _zipWithIndex(keys.rdd, True)
        new_schema = to_spark_schema(to_schema(df.schema).extract(cols) + f'{_PARTITION_DUMMY_KEY}:long')
        idx = session.createDataFrame(krdd, new_schema)
        if num <= 0:
            idx = idx.persist()
            num = idx.count()
        idf = df.alias('df').join(idx.alias('idx'), on=cols, how='inner').select(_PARTITION_DUMMY_KEY, *['df.' + x for x in df.schema.names])

        def _to_kv(rows: Iterable[Any]) -> Iterable[Any]:
            for row in rows:
                yield (row[0], row[1:])
        rdd = idf.rdd.mapPartitions(_to_kv).partitionBy(num, lambda k: k).mapPartitions(_to_rows)
        return session.createDataFrame(rdd, df.schema)