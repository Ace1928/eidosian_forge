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
def pandas_udf_can_accept(schema: Schema, is_input: bool) -> bool:
    try:
        if not is_input and any((pa.types.is_struct(t) for t in schema.types)):
            return False
        to_arrow_schema(from_arrow_schema(schema.pa_schema))
        return True
    except Exception:
        return False