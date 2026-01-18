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
def get_attribute_from_bases(tp: type[Any] | tuple[type[Any], ...], name: str) -> Any:
    """Get the attribute from the next class in the MRO that has it,
    aiming to simulate calling the method on the actual class.

    The reason for iterating over the mro instead of just getting
    the attribute (which would do that for us) is to support TypedDict,
    which lacks a real __mro__, but can have a virtual one constructed
    from its bases (as done here).

    Args:
        tp: The type or class to search for the attribute. If a tuple, this is treated as a set of base classes.
        name: The name of the attribute to retrieve.

    Returns:
        Any: The attribute value, if found.

    Raises:
        AttributeError: If the attribute is not found in any class in the MRO.
    """
    if isinstance(tp, tuple):
        for base in mro_for_bases(tp):
            attribute = base.__dict__.get(name, _sentinel)
            if attribute is not _sentinel:
                attribute_get = getattr(attribute, '__get__', None)
                if attribute_get is not None:
                    return attribute_get(None, tp)
                return attribute
        raise AttributeError(f'{name} not found in {tp}')
    else:
        try:
            return getattr(tp, name)
        except AttributeError:
            return get_attribute_from_bases(mro(tp), name)