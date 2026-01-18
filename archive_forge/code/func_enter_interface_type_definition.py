from operator import attrgetter
from typing import Any, Collection
from ...error import GraphQLError
from ...language import (
from ...pyutils import group_by
from . import SDLValidationRule
def enter_interface_type_definition(self, node: InterfaceTypeDefinitionNode, *_args: Any) -> VisitorAction:
    return self.check_arg_uniqueness_per_field(node.name, node.fields)