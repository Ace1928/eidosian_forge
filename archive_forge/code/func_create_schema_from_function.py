from __future__ import annotations
import inspect
import uuid
import warnings
from abc import abstractmethod
from functools import partial
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain_core.callbacks import (
from langchain_core.callbacks.manager import (
from langchain_core.load.serializable import Serializable
from langchain_core.prompts import (
from langchain_core.pydantic_v1 import (
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
from langchain_core.runnables.config import run_in_executor
def create_schema_from_function(model_name: str, func: Callable) -> Type[BaseModel]:
    """Create a pydantic schema from a function's signature.
    Args:
        model_name: Name to assign to the generated pydandic schema
        func: Function to generate the schema from
    Returns:
        A pydantic model with the same arguments as the function
    """
    validated = validate_arguments(func, config=_SchemaConfig)
    inferred_model = validated.model
    if 'run_manager' in inferred_model.__fields__:
        del inferred_model.__fields__['run_manager']
    if 'callbacks' in inferred_model.__fields__:
        del inferred_model.__fields__['callbacks']
    valid_properties = _get_filtered_args(inferred_model, func)
    return _create_subset_model(f'{model_name}Schema', inferred_model, list(valid_properties))