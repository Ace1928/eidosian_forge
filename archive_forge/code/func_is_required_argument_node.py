from typing import cast, Any, Dict, List, Union
from ...error import GraphQLError
from ...language import (
from ...type import GraphQLArgument, is_required_argument, is_type, specified_directives
from . import ASTValidationRule, SDLValidationContext, ValidationContext
def is_required_argument_node(arg: InputValueDefinitionNode) -> bool:
    return isinstance(arg.type, NonNullTypeNode) and arg.default_value is None