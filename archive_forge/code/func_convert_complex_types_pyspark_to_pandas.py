import datetime as dt
import decimal
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.store.artifact.utils.models import get_model_name_and_version
from mlflow.types import DataType, ParamSchema, ParamSpec, Schema, TensorSpec
from mlflow.types.schema import Array, Map, Object, Property
from mlflow.types.utils import (
from mlflow.utils.annotations import experimental
from mlflow.utils.proto_json_utils import (
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
def convert_complex_types_pyspark_to_pandas(value, dataType):
    type_mapping = {IntegerType: lambda v: np.int32(v), ShortType: lambda v: np.int16(v), FloatType: lambda v: np.float32(v), DateType: lambda v: v.strftime('%Y-%m-%d'), TimestampType: lambda v: v.strftime('%Y-%m-%d %H:%M:%S.%f'), BinaryType: lambda v: np.bytes_(v)}
    if value is None:
        return None
    if isinstance(dataType, StructType):
        return {field.name: convert_complex_types_pyspark_to_pandas(value[field.name], field.dataType) for field in dataType.fields}
    elif isinstance(dataType, ArrayType):
        return [convert_complex_types_pyspark_to_pandas(elem, dataType.elementType) for elem in value]
    converter = type_mapping.get(type(dataType))
    if converter:
        return converter(value)
    return value