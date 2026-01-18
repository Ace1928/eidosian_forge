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
@classmethod
def enforce_param_datatype(cls, name, value, dtype: DataType):
    """
        Enforce the value matches the data type.

        The following type conversions are allowed:

        1. int -> long, float, double
        2. long -> float, double
        3. float -> double
        4. any -> datetime (try conversion)

        Any other type mismatch will raise error.

        Args:
            name: parameter name
            value: parameter value
            t: expected data type
        """
    if value is None:
        return
    if dtype == DataType.datetime:
        try:
            datetime_value = np.datetime64(value).item()
            if isinstance(datetime_value, int):
                raise MlflowException.invalid_parameter_value(f'Invalid value for param {name}, it should be convertible to datetime.date/datetime, got {value}')
            return datetime_value
        except ValueError as e:
            raise MlflowException.invalid_parameter_value(f'Failed to convert value {value} from type {type(value).__name__} to {dtype} for param {name}') from e
    if not np.isscalar(value):
        raise MlflowException.invalid_parameter_value(f'Value should be a scalar for param {name}, got {value}')
    if getattr(DataType, f'is_{dtype.name}')(value):
        return DataType[dtype.name].to_python()(value)
    if DataType.is_integer(value) and dtype in (DataType.long, DataType.float, DataType.double) or (DataType.is_long(value) and dtype in (DataType.float, DataType.double)) or (DataType.is_float(value) and dtype == DataType.double):
        try:
            return DataType[dtype.name].to_python()(value)
        except ValueError as e:
            raise MlflowException.invalid_parameter_value(f'Failed to convert value {value} from type {type(value).__name__} to {dtype} for param {name}') from e
    raise MlflowException.invalid_parameter_value(f'Incompatible types for param {name}. Can not safely convert {type(value).__name__} to {dtype}.')