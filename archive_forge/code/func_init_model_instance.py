from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, overload
from . import validator
from .config import Extra
from .errors import ConfigError
from .main import BaseModel, create_model
from .typing import get_all_type_hints
from .utils import to_camel
def init_model_instance(self, *args: Any, **kwargs: Any) -> BaseModel:
    values = self.build_values(args, kwargs)
    return self.model(**values)