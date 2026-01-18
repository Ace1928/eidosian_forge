from ..language import visitor_meta
from ..type.definition import (GraphQLInputObjectType, GraphQLList,
from .get_field_def import get_field_def
from .type_from_ast import type_from_ast
def enter_ListValue(self, node):
    list_type = get_nullable_type(self.get_input_type())
    self._input_type_stack.append(list_type.of_type if isinstance(list_type, GraphQLList) else None)