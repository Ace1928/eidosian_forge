from operator import attrgetter
from typing import Any, Collection
from ...error import GraphQLError
from ...language import (
from ...pyutils import group_by
from . import SDLValidationRule
def check_arg_uniqueness_per_field(self, name: NameNode, fields: Collection[FieldDefinitionNode]) -> VisitorAction:
    type_name = name.value
    for field_def in fields:
        field_name = field_def.name.value
        argument_nodes = field_def.arguments or ()
        self.check_arg_uniqueness(f'{type_name}.{field_name}', argument_nodes)
    return SKIP