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
def get_caller_frame_info() -> Tuple[Optional[str], bool]:
    """
    Used inside a function to check whether it was called globally

    Will only work against non-compiled code, therefore used only in pydantic.generics

    :returns Tuple[module_name, called_globally]
    """
    try:
        previous_caller_frame = sys._getframe(2)
    except ValueError as e:
        raise RuntimeError('This function must be used inside another function') from e
    except AttributeError:
        return (None, False)
    frame_globals = previous_caller_frame.f_globals
    return (frame_globals.get('__name__'), previous_caller_frame.f_locals is frame_globals)