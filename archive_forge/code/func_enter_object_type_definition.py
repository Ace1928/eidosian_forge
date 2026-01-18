from operator import attrgetter
from typing import Any, Collection
from ...error import GraphQLError
from ...language import (
from ...pyutils import group_by
from . import SDLValidationRule
def enter_object_type_definition(self, node: ObjectTypeDefinitionNode, *_args: Any) -> VisitorAction:
    return self.check_arg_uniqueness_per_field(node.name, node.fields)