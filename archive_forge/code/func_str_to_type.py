import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def str_to_type(s: str, expected_base_type: Optional[type]=None, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None) -> type:
    """Given a string expression, find the first/last type from all import libraries.
    If the expression contains `.`, it's supposed to be a relative or full path of
    the type including modules.

    :param s: type expression, for example `triad.utils.iter.Slicer` or `str`
    :param expected_base_type: base class type that must satisfy, defaults to None
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :raises TypeError: unable to find a matching type

    :return: found type
    """
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    try:
        obj = str_to_object(s, global_vars, local_vars)
    except ValueError:
        raise TypeError(f'{s} is not a type')
    assert_or_throw(isinstance(obj, type), TypeError(f'{obj} is not a type'))
    assert_or_throw(expected_base_type is None or issubclass(obj, expected_base_type), TypeError(f'{obj} is not a subtype of {expected_base_type}'))
    return obj