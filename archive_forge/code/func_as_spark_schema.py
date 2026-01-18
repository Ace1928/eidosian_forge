import builtins
import datetime as dt
import importlib.util
import json
import string
import warnings
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
def as_spark_schema(self):
    """Convert to Spark schema. If this schema is a single unnamed column, it is converted
        directly the corresponding spark data type, otherwise it's returned as a struct (missing
        column names are filled with an integer sequence).
        Unsupported by TensorSpec.
        """
    if self.is_tensor_spec():
        raise MlflowException('TensorSpec cannot be converted to spark dataframe')
    if len(self.inputs) == 1 and self.inputs[0].name is None:
        return self.inputs[0].type.to_spark()
    from pyspark.sql.types import StructField, StructType
    return StructType([StructField(name=col.name or str(i), dataType=col.type.to_spark(), nullable=not col.required) for i, col in enumerate(self.inputs)])