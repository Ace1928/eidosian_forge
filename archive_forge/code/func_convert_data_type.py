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
def convert_data_type(data, spec):
    """
    Convert input data to the type specified in the spec.

    This method converts data into numpy array for backwards compatibility.

    Args:
        data: Input data.
        spec: ColSpec or TensorSpec.
    """
    import numpy as np
    from mlflow.models.utils import _enforce_array, _enforce_map, _enforce_object
    from mlflow.types.schema import Array, ColSpec, DataType, Map, Object, TensorSpec
    try:
        if spec is None:
            return np.array(data)
        if isinstance(spec, TensorSpec):
            return np.array(data, dtype=spec.type)
        if isinstance(spec, ColSpec):
            if isinstance(spec.type, DataType):
                return np.array(data, spec.type.to_numpy()) if isinstance(data, (list, np.ndarray)) else np.array([data], spec.type.to_numpy())[0]
            elif isinstance(spec.type, Array):
                return np.array(_enforce_array(data, spec.type))
            elif isinstance(spec.type, Object):
                return _enforce_object(data, spec.type)
            elif isinstance(spec.type, Map):
                return _enforce_map(data, spec.type)
    except MlflowException as e:
        raise MlflowInvalidInputException(e.message)
    except Exception as ex:
        raise MlflowInvalidInputException(f'{ex}')
    raise MlflowInvalidInputException(f'Failed to convert data type for data `{data}` with spec `{spec}`.')