import pickle
from typing import Any, Iterable, List, Tuple, Optional
import pandas as pd
import pyarrow as pa
import pyspark
import pyspark.sql as ps
import pyspark.sql.types as pt
from packaging import version
from pyarrow.types import is_list, is_struct, is_timestamp
from pyspark.sql.pandas.types import (
from triad.collections import Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP, cast_pa_table
from triad.utils.schema import quote_name
import fugue.api as fa
from fugue import DataFrame
from .misc import is_spark_dataframe
def pd_to_spark_df(session: ps.SparkSession, df: pd.DataFrame, schema: Any=None) -> ps.DataFrame:
    if _PYSPARK_ARROW_FRIENDLY:
        return session.createDataFrame(df, schema=schema)
    else:
        return session.createDataFrame(df.astype(object), schema=schema)