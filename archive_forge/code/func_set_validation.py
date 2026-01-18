import copy
import dataclasses
import sys
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generator, Optional, Type, TypeVar, Union, overload
from typing_extensions import dataclass_transform
from .class_validators import gather_all_validators
from .config import BaseConfig, ConfigDict, Extra, get_config
from .error_wrappers import ValidationError
from .errors import DataclassTypeError
from .fields import Field, FieldInfo, Required, Undefined
from .main import create_model, validate_model
from .utils import ClassAttribute
@contextmanager
def set_validation(cls: Type['DataclassT'], value: bool) -> Generator[Type['DataclassT'], None, None]:
    original_run_validation = cls.__pydantic_run_validation__
    try:
        cls.__pydantic_run_validation__ = value
        yield cls
    finally:
        cls.__pydantic_run_validation__ = original_run_validation