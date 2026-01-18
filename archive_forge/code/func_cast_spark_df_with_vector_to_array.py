import re
from functools import reduce
from typing import Set, Union
from pyspark.ml.base import Transformer
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import types as t
def cast_spark_df_with_vector_to_array(input_spark_df):
    """
    Finds columns of vector type in a spark dataframe and
    casts them to array<double> type.

    Args:
        input_spark_df:

    Returns:
        A spark dataframe with vector columns transformed to array<double> type
    """
    vector_type_columns = [_field.name for _field in input_spark_df.schema if isinstance(_field.dataType, VectorUDT)]
    return reduce(lambda df, vector_col: df.withColumn(vector_col, vector_to_array(vector_col)), vector_type_columns, input_spark_df)