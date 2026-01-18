import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def as_type(obj: Any, target: type) -> Any:
    """Convert `obj` into `target` type

    :param obj: input object
    :param target: target type

    :return: object in the target type
    """
    if issubclass(type(obj), target):
        return obj
    if target == bool:
        return to_bool(obj)
    if target == datetime.datetime:
        return to_datetime(obj)
    if target == datetime.timedelta:
        return to_timedelta(obj)
    return target(obj)