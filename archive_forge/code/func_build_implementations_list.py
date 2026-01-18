from itertools import chain
from typing import cast, Callable, Collection, Dict, List, Union
from ..language import DirectiveLocation, parse_value
from ..pyutils import inspect, Undefined
from ..type import (
from .get_introspection_query import (
from .value_from_ast import value_from_ast
def build_implementations_list(implementing_introspection: Union[IntrospectionObjectType, IntrospectionInterfaceType]) -> List[GraphQLInterfaceType]:
    maybe_interfaces = implementing_introspection.get('interfaces')
    if maybe_interfaces is None:
        if implementing_introspection['kind'] == TypeKind.INTERFACE.name:
            return []
        raise TypeError(f'Introspection result missing interfaces: {inspect(implementing_introspection)}.')
    interfaces = cast(Collection[IntrospectionInterfaceType], maybe_interfaces)
    return [get_interface_type(interface) for interface in interfaces]