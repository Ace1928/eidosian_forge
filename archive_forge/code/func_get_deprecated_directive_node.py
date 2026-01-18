from operator import attrgetter, itemgetter
from typing import (
from ..error import GraphQLError
from ..pyutils import inspect
from ..language import (
from .definition import (
from ..utilities.type_comparators import is_equal_type, is_type_sub_type_of
from .directives import is_directive, GraphQLDeprecatedDirective
from .introspection import is_introspection_type
from .schema import GraphQLSchema, assert_schema
def get_deprecated_directive_node(definition_node: Optional[Union[InputValueDefinitionNode]]) -> Optional[DirectiveNode]:
    directives = definition_node and definition_node.directives
    if directives:
        for directive in directives:
            if directive.name.value == GraphQLDeprecatedDirective.name:
                return directive
    return None