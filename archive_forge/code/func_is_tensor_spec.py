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
def is_tensor_spec(self) -> bool:
    """Return true iff this schema is specified using TensorSpec"""
    return self.inputs and isinstance(self.inputs[0], TensorSpec)