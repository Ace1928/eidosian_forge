from __future__ import annotations
import datetime as dt
import os
import sys
from typing import Any, Iterable
import numpy as np
import param
def is_parameterized(obj) -> bool:
    """
    Whether an object is a Parameterized class or instance.
    """
    return isinstance(obj, param.Parameterized) or (isinstance(obj, type) and issubclass(obj, param.Parameterized))