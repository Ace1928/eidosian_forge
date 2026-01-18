import sys
import types
import typing
from typing import (
from weakref import WeakKeyDictionary, WeakValueDictionary
from typing_extensions import Annotated, Literal as ExtLiteral
from .class_validators import gather_all_validators
from .fields import DeferredType
from .main import BaseModel, create_model
from .types import JsonWrapper
from .typing import display_as_type, get_all_type_hints, get_args, get_origin, typing_base
from .utils import all_identical, lenient_issubclass
def build_base_model(base_model: Type[GenericModel], mapped_types: Parametrization) -> Iterator[Type[GenericModel]]:
    base_parameters = tuple((mapped_types[param] for param in base_model.__parameters__))
    parameterized_base = base_model.__class_getitem__(base_parameters)
    if parameterized_base is base_model or parameterized_base is cls:
        return
    yield parameterized_base