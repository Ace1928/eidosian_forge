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
def pandas_types(self) -> List[np.dtype]:
    """Convenience shortcut to get the datatypes as pandas types. Unsupported by TensorSpec."""
    if self.is_tensor_spec():
        raise MlflowException('TensorSpec only supports numpy types, use numpy_types() instead')
    if all((isinstance(x.type, DataType) for x in self.inputs)):
        return [x.type.to_pandas() for x in self.inputs]
    raise MlflowException('Failed to get pandas types as some of the inputs types are not DataType.')