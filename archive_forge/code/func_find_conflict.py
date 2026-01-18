from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def find_conflict(context: ValidationContext, cached_fields_and_fragment_names: Dict, compared_fragment_pairs: 'PairSet', parent_fields_are_mutually_exclusive: bool, response_name: str, field1: NodeAndDef, field2: NodeAndDef) -> Optional[Conflict]:
    """Find conflict.

    Determines if there is a conflict between two particular fields, including comparing
    their sub-fields.
    """
    parent_type1, node1, def1 = field1
    parent_type2, node2, def2 = field2
    are_mutually_exclusive = parent_fields_are_mutually_exclusive or (parent_type1 != parent_type2 and is_object_type(parent_type1) and is_object_type(parent_type2))
    type1 = cast(Optional[GraphQLOutputType], def1 and def1.type)
    type2 = cast(Optional[GraphQLOutputType], def2 and def2.type)
    if not are_mutually_exclusive:
        name1 = node1.name.value
        name2 = node2.name.value
        if name1 != name2:
            return ((response_name, f"'{name1}' and '{name2}' are different fields"), [node1], [node2])
        if stringify_arguments(node1) != stringify_arguments(node2):
            return ((response_name, 'they have differing arguments'), [node1], [node2])
    if type1 and type2 and do_types_conflict(type1, type2):
        return ((response_name, f"they return conflicting types '{type1}' and '{type2}'"), [node1], [node2])
    selection_set1 = node1.selection_set
    selection_set2 = node2.selection_set
    if selection_set1 and selection_set2:
        conflicts = find_conflicts_between_sub_selection_sets(context, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, get_named_type(type1), selection_set1, get_named_type(type2), selection_set2)
        return subfield_conflicts(conflicts, response_name, node1, node2)
    return None