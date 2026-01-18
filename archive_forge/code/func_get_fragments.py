from typing import Callable, Dict, List, Optional, Union, Tuple
from graphql import GraphQLError
from graphql.validation import ValidationContext, ValidationRule
from graphql.language import (
from ..utils.is_introspection_key import is_introspection_key
def get_fragments(definitions: Tuple[DefinitionNode, ...]) -> Dict[str, FragmentDefinitionNode]:
    fragments = {}
    for definition in definitions:
        if isinstance(definition, FragmentDefinitionNode):
            fragments[definition.name.value] = definition
    return fragments