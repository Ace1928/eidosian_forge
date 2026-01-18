from asyncio import ensure_future, gather
from collections.abc import Mapping
from inspect import isawaitable
from typing import (
from ..error import GraphQLError, GraphQLFormattedError, located_error
from ..language import (
from ..pyutils import (
from ..type import (
from .collect_fields import collect_fields, collect_sub_fields
from .middleware import MiddlewareManager
from .values import get_argument_values, get_variable_values
def get_typename(value: Any) -> Optional[str]:
    """Get the ``__typename`` property of the given value."""
    if isinstance(value, Mapping):
        return value.get('__typename')
    for cls in value.__class__.__mro__:
        __typename = getattr(value, f'_{cls.__name__}__typename', None)
        if __typename:
            return __typename
    return None