from operator import attrgetter
from typing import Any, Collection
from ...error import GraphQLError
from ...language import (
from ...pyutils import group_by
from . import SDLValidationRule
def enter_directive_definition(self, node: DirectiveDefinitionNode, *_args: Any) -> VisitorAction:
    return self.check_arg_uniqueness(f'@{node.name.value}', node.arguments)