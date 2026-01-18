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
def get_all_types(self):
    types = [self.to_numpy(), self.to_pandas(), self.to_python()]
    if importlib.util.find_spec('pyspark') is not None:
        types.append(self.to_spark())
    if self.name == 'datetime':
        types.extend([np.datetime64, dt.datetime])
    if self.name == 'binary':
        types.extend([bytearray])
    return types