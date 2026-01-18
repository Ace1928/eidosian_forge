from __future__ import annotations
import enum
import threading
from abc import abstractmethod
from functools import wraps
from typing import (
from weakref import WeakValueDictionary
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
def make_options_spec(spec: Union[ConfigurableFieldSingleOption, ConfigurableFieldMultiOption], description: Optional[str]) -> ConfigurableFieldSpec:
    """Make a ConfigurableFieldSpec for a ConfigurableFieldSingleOption or
    ConfigurableFieldMultiOption."""
    with _enums_for_spec_lock:
        if (enum := _enums_for_spec.get(spec)):
            pass
        else:
            enum = StrEnum(spec.name or spec.id, ((v, v) for v in list(spec.options.keys())))
            _enums_for_spec[spec] = cast(Type[StrEnum], enum)
    if isinstance(spec, ConfigurableFieldSingleOption):
        return ConfigurableFieldSpec(id=spec.id, name=spec.name, description=spec.description or description, annotation=enum, default=spec.default, is_shared=spec.is_shared)
    else:
        return ConfigurableFieldSpec(id=spec.id, name=spec.name, description=spec.description or description, annotation=Sequence[enum], default=spec.default, is_shared=spec.is_shared)