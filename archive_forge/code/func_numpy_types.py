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
def numpy_types(self) -> List[np.dtype]:
    """Convenience shortcut to get the datatypes as numpy types."""
    if self.is_tensor_spec():
        return [x.type for x in self.inputs]
    if all((isinstance(x.type, DataType) for x in self.inputs)):
        return [x.type.to_numpy() for x in self.inputs]
    raise MlflowException('Failed to get numpy types as some of the inputs types are not DataType.')