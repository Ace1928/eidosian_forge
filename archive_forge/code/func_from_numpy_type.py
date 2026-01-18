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
def from_numpy_type(cls, np_type):
    return next((v for v in cls._member_map_.values() if v.to_numpy() == np_type), None)