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
def default_field_resolver(source: Any, info: GraphQLResolveInfo, **args: Any) -> Any:
    """Default field resolver.

    If a resolve function is not given, then a default resolve behavior is used which
    takes the property of the source object of the same name as the field and returns
    it as the result, or if it's a function, returns the result of calling that function
    while passing along args and context.

    For dictionaries, the field names are used as keys, for all other objects they are
    used as attribute names.
    """
    field_name = info.field_name
    value = source.get(field_name) if isinstance(source, Mapping) else getattr(source, field_name, None)
    if callable(value):
        return value(info, **args)
    return value