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
def ensure_valid_runtime_type(self, runtime_type_name: Any, return_type: GraphQLAbstractType, field_nodes: List[FieldNode], info: GraphQLResolveInfo, result: Any) -> GraphQLObjectType:
    if runtime_type_name is None:
        raise GraphQLError(f"Abstract type '{return_type.name}' must resolve to an Object type at runtime for field '{info.parent_type.name}.{info.field_name}'. Either the '{return_type.name}' type should provide a 'resolve_type' function or each possible type should provide an 'is_type_of' function.", field_nodes)
    if is_object_type(runtime_type_name):
        raise GraphQLError('Support for returning GraphQLObjectType from resolve_type was removed in GraphQL-core 3.2, please return type name instead.')
    if not isinstance(runtime_type_name, str):
        raise GraphQLError(f"Abstract type '{return_type.name}' must resolve to an Object type at runtime for field '{info.parent_type.name}.{info.field_name}' with value {inspect(result)}, received '{inspect(runtime_type_name)}'.", field_nodes)
    runtime_type = self.schema.get_type(runtime_type_name)
    if runtime_type is None:
        raise GraphQLError(f"Abstract type '{return_type.name}' was resolved to a type '{runtime_type_name}' that does not exist inside the schema.", field_nodes)
    if not is_object_type(runtime_type):
        raise GraphQLError(f"Abstract type '{return_type.name}' was resolved to a non-object type '{runtime_type_name}'.", field_nodes)
    runtime_type = cast(GraphQLObjectType, runtime_type)
    if not self.schema.is_sub_type(return_type, runtime_type):
        raise GraphQLError(f"Runtime Object type '{runtime_type.name}' is not a possible type for '{return_type.name}'.", field_nodes)
    return runtime_type