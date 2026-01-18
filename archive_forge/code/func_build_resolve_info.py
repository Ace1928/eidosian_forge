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
def build_resolve_info(self, field_def: GraphQLField, field_nodes: List[FieldNode], parent_type: GraphQLObjectType, path: Path) -> GraphQLResolveInfo:
    """Build the GraphQLResolveInfo object.

        For internal use only."""
    return GraphQLResolveInfo(field_nodes[0].name.value, field_nodes, field_def.type, parent_type, path, self.schema, self.fragments, self.root_value, self.operation, self.variable_values, self.context_value, self.is_awaitable)