import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, ForwardRef, Optional, Tuple, Type, Union
from typing_extensions import Literal, Protocol
from .typing import AnyArgTCallable, AnyCallable
from .utils import GetterDict
from .version import compiled
@classmethod
def prepare_field(cls, field: 'ModelField') -> None:
    """
        Optional hook to check or modify fields during model creation.
        """
    pass