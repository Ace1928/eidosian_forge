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
def collect_subfields(self, return_type: GraphQLObjectType, field_nodes: List[FieldNode]) -> Dict[str, List[FieldNode]]:
    """Collect subfields.

        A cached collection of relevant subfields with regard to the return type is
        kept in the execution context as ``_subfields_cache``. This ensures the
        subfields are not repeatedly calculated, which saves overhead when resolving
        lists of values.
        """
    cache = self._subfields_cache
    key = (return_type, id(field_nodes[0])) if len(field_nodes) == 1 else tuple((return_type, *map(id, field_nodes)))
    sub_field_nodes = cache.get(key)
    if sub_field_nodes is None:
        sub_field_nodes = collect_sub_fields(self.schema, self.fragments, self.variable_values, return_type, field_nodes)
        cache[key] = sub_field_nodes
    return sub_field_nodes