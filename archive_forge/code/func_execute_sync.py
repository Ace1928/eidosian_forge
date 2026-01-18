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
def execute_sync(schema: GraphQLSchema, document: DocumentNode, root_value: Any=None, context_value: Any=None, variable_values: Optional[Dict[str, Any]]=None, operation_name: Optional[str]=None, field_resolver: Optional[GraphQLFieldResolver]=None, type_resolver: Optional[GraphQLTypeResolver]=None, middleware: Optional[Middleware]=None, execution_context_class: Optional[Type['ExecutionContext']]=None, check_sync: bool=False) -> ExecutionResult:
    """Execute a GraphQL operation synchronously.

    Also implements the "Executing requests" section of the GraphQL specification.

    However, it guarantees to complete synchronously (or throw an error) assuming
    that all field resolvers are also synchronous.

    Set check_sync to True to still run checks that no awaitable values are returned.
    """
    is_awaitable = check_sync if callable(check_sync) else None if check_sync else assume_not_awaitable
    result = execute(schema, document, root_value, context_value, variable_values, operation_name, field_resolver, type_resolver, None, middleware, execution_context_class, is_awaitable)
    if isawaitable(result):
        ensure_future(cast(Awaitable[ExecutionResult], result)).cancel()
        raise RuntimeError('GraphQL execution failed to complete synchronously.')
    return cast(ExecutionResult, result)