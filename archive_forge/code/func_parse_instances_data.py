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
def parse_instances_data(data, schema=None):
    import numpy as np
    from mlflow.types.schema import Array
    if 'instances' not in data:
        raise MlflowInvalidInputException('Expecting data to have `instances` as key.')
    data = data['instances']
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        data_dict = defaultdict(list)
        types_dict = schema.input_dict() if schema and schema.has_input_names() else {}
        for item in data:
            for col, v in item.items():
                data_dict[col].append(convert_data_type(v, types_dict.get(col)))
        data = {col: np.array(v) for col, v in data_dict.items()}
    else:
        data = _cast_schema_type(data, schema)
    if isinstance(data, dict):
        check_data = {k: v for k, v in data.items() if isinstance(v, (list, np.ndarray))}
        if schema and schema.has_input_names():
            required_cols = schema.required_input_names()
            check_cols = {col for col, spec in schema.input_dict().items() if not isinstance(spec.type, Array)}
            check_cols = list(set(required_cols) & check_cols & set(check_data.keys()))
        else:
            check_cols = list(check_data.keys())
        if check_cols:
            expected_len = len(check_data[check_cols[0]])
            if not all((len(check_data[col]) == expected_len for col in check_cols[1:])):
                raise MlflowInvalidInputException('The length of values for each input/column name are not the same')
    return data