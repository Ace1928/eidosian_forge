from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, overload
from . import validator
from .config import Extra
from .errors import ConfigError
from .main import BaseModel, create_model
from .typing import get_all_type_hints
from .utils import to_camel
@validator(V_POSITIONAL_ONLY_NAME, check_fields=False, allow_reuse=True)
def check_positional_only(cls, v: Optional[List[str]]) -> None:
    if v is None:
        return
    plural = '' if len(v) == 1 else 's'
    keys = ', '.join(map(repr, v))
    raise TypeError(f'positional-only argument{plural} passed as keyword argument{plural}: {keys}')