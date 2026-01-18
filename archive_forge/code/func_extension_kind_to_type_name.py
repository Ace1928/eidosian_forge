import re
from functools import partial
from typing import Any, Optional
from ...error import GraphQLError
from ...language import TypeDefinitionNode, TypeExtensionNode
from ...pyutils import did_you_mean, inspect, suggestion_list
from ...type import (
from . import SDLValidationContext, SDLValidationRule
def extension_kind_to_type_name(kind: str) -> str:
    return _type_names_for_extension_kinds.get(kind, 'unknown type')