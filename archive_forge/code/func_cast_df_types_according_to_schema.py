import base64
import datetime
import importlib
import json
import os
from collections import defaultdict
from copy import deepcopy
from functools import partial
from json import JSONEncoder
from typing import Any, Dict, Optional
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.json_format import MessageToJson, ParseDict
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
def cast_df_types_according_to_schema(pdf, schema):
    import numpy as np
    from mlflow.models.utils import _enforce_array, _enforce_map, _enforce_object
    from mlflow.types.schema import Array, DataType, Map, Object
    actual_cols = set(pdf.columns)
    if schema.has_input_names():
        dtype_list = zip(schema.input_names(), schema.input_types())
    elif schema.is_tensor_spec() and len(schema.input_types()) == 1:
        dtype_list = zip(actual_cols, [schema.input_types()[0] for _ in actual_cols])
    else:
        n = min(len(schema.input_types()), len(pdf.columns))
        dtype_list = zip(pdf.columns[:n], schema.input_types()[:n])
    for col_name, col_type_spec in dtype_list:
        if isinstance(col_type_spec, DataType):
            col_type = col_type_spec.to_pandas()
        else:
            col_type = col_type_spec
        if col_name in actual_cols:
            try:
                if isinstance(col_type_spec, DataType) and col_type_spec == DataType.binary:
                    pdf[col_name] = pdf[col_name].map(lambda x: base64.decodebytes(bytes(x, 'utf8')))
                elif col_type == np.dtype(bytes):
                    pdf[col_name] = pdf[col_name].map(lambda x: bytes(x, 'utf8'))
                elif schema.is_tensor_spec() and isinstance(pdf[col_name].iloc[0], list):
                    pass
                elif isinstance(col_type_spec, Array):
                    pdf[col_name] = pdf[col_name].map(lambda x: _enforce_array(x, col_type_spec))
                elif isinstance(col_type_spec, Object):
                    pdf[col_name] = pdf[col_name].map(lambda x: _enforce_object(x, col_type_spec))
                elif isinstance(col_type_spec, Map):
                    pdf[col_name] = pdf[col_name].map(lambda x: _enforce_map(x, col_type_spec))
                else:
                    pdf[col_name] = pdf[col_name].astype(col_type, copy=False)
            except Exception as ex:
                raise MlflowFailedTypeConversion(col_name, col_type, ex)
    return pdf