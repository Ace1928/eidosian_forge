from typing import Any, Collection, List, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import introspection_types, specified_scalar_types
from ...pyutils import did_you_mean, suggestion_list
from . import ASTValidationRule, ValidationContext, SDLValidationContext
def is_sdl_node(value: Union[Node, Collection[Node], None]) -> bool:
    return value is not None and (not isinstance(value, list)) and (is_type_system_definition_node(cast(Node, value)) or is_type_system_extension_node(cast(Node, value)))