from itertools import chain
from typing import cast, Callable, Collection, Dict, List, Union
from ..language import DirectiveLocation, parse_value
from ..pyutils import inspect, Undefined
from ..type import (
from .get_introspection_query import (
from .value_from_ast import value_from_ast
def build_field(field_introspection: IntrospectionField) -> GraphQLField:
    type_introspection = cast(IntrospectionType, field_introspection['type'])
    type_ = get_type(type_introspection)
    if not is_output_type(type_):
        raise TypeError(f'Introspection must provide output type for fields, but received: {inspect(type_)}.')
    type_ = cast(GraphQLOutputType, type_)
    args_introspection = field_introspection.get('args')
    if args_introspection is None:
        raise TypeError(f'Introspection result missing field args: {inspect(field_introspection)}.')
    return GraphQLField(type_, args=build_argument_def_map(args_introspection), description=field_introspection.get('description'), deprecation_reason=field_introspection.get('deprecationReason'))