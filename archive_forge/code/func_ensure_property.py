from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
def ensure_property(f: Any) -> Any:
    """Ensure that a function is a `property` or `cached_property`, or is a valid descriptor.

    Args:
        f: The function to check.

    Returns:
        The function, or a `property` or `cached_property` instance wrapping the function.
    """
    if ismethoddescriptor(f) or isdatadescriptor(f):
        return f
    else:
        return property(f)