from collections import defaultdict
from typing import Any, Dict
from ...error import GraphQLError
from ...language import NameNode, ObjectTypeDefinitionNode, VisitorAction, SKIP
from ...type import is_object_type, is_interface_type, is_input_object_type
from . import SDLValidationContext, SDLValidationRule
def has_field(type_: Any, field_name: str) -> bool:
    if is_object_type(type_) or is_interface_type(type_) or is_input_object_type(type_):
        return field_name in type_.fields
    return False