import fnmatch
import importlib
import inspect
import sys
from dataclasses import dataclass
from enum import Enum
from functools import partial
from inspect import signature
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Type, TypeVar, Union
from torch import nn
from .._internally_replaced_utils import load_state_dict_from_url
def get_model_weights(name: Union[Callable, str]) -> Type[WeightsEnum]:
    """
    Returns the weights enum class associated to the given model.

    Args:
        name (callable or str): The model builder function or the name under which it is registered.

    Returns:
        weights_enum (WeightsEnum): The weights enum class associated with the model.
    """
    model = get_model_builder(name) if isinstance(name, str) else name
    return _get_enum_from_fn(model)