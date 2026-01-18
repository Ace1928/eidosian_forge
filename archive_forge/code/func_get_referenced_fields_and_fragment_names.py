from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def get_referenced_fields_and_fragment_names(context: ValidationContext, cached_fields_and_fragment_names: Dict, fragment: FragmentDefinitionNode) -> Tuple[NodeAndDefCollection, List[str]]:
    """Get referenced fields and nested fragment names

    Given a reference to a fragment, return the represented collection of fields as well
    as a list of nested fragment names referenced via fragment spreads.
    """
    cached = cached_fields_and_fragment_names.get(fragment.selection_set)
    if cached:
        return cached
    fragment_type = type_from_ast(context.schema, fragment.type_condition)
    return get_fields_and_fragment_names(context, cached_fields_and_fragment_names, fragment_type, fragment.selection_set)