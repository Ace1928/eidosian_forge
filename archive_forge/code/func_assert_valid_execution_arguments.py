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
def assert_valid_execution_arguments(schema: GraphQLSchema, document: DocumentNode, raw_variable_values: Optional[Dict[str, Any]]=None) -> None:
    """Check that the arguments are acceptable.

    Essential assertions before executing to provide developer feedback for improper use
    of the GraphQL library.

    For internal use only.
    """
    if not document:
        raise TypeError('Must provide document.')
    assert_valid_schema(schema)
    if not (raw_variable_values is None or isinstance(raw_variable_values, dict)):
        raise TypeError('Variable values must be provided as a dictionary with variable names as keys. Perhaps look to see if an unparsed JSON string was provided.')