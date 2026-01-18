from asyncio import ensure_future
from inspect import isawaitable
from typing import Any, Awaitable, Callable, Dict, Optional, Union, Type, cast
from .error import GraphQLError
from .execution import execute, ExecutionResult, ExecutionContext, Middleware
from .language import parse, Source
from .pyutils import AwaitableOrValue
from .type import (
def graphql_impl(schema: GraphQLSchema, source: Union[str, Source], root_value: Any, context_value: Any, variable_values: Optional[Dict[str, Any]], operation_name: Optional[str], field_resolver: Optional[GraphQLFieldResolver], type_resolver: Optional[GraphQLTypeResolver], middleware: Optional[Middleware], execution_context_class: Optional[Type[ExecutionContext]], is_awaitable: Optional[Callable[[Any], bool]]) -> AwaitableOrValue[ExecutionResult]:
    """Execute a query, return asynchronously only if necessary."""
    schema_validation_errors = validate_schema(schema)
    if schema_validation_errors:
        return ExecutionResult(data=None, errors=schema_validation_errors)
    try:
        document = parse(source)
    except GraphQLError as error:
        return ExecutionResult(data=None, errors=[error])
    from .validation import validate
    validation_errors = validate(schema, document)
    if validation_errors:
        return ExecutionResult(data=None, errors=validation_errors)
    return execute(schema, document, root_value, context_value, variable_values, operation_name, field_resolver, type_resolver, None, middleware, execution_context_class, is_awaitable)