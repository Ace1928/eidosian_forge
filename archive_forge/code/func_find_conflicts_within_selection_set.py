from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def find_conflicts_within_selection_set(context: ValidationContext, cached_fields_and_fragment_names: Dict, compared_fragment_pairs: 'PairSet', parent_type: Optional[GraphQLNamedType], selection_set: SelectionSetNode) -> List[Conflict]:
    """Find conflicts within selection set.

    Find all conflicts found "within" a selection set, including those found via
    spreading in fragments.

    Called when visiting each SelectionSet in the GraphQL Document.
    """
    conflicts: List[Conflict] = []
    field_map, fragment_names = get_fields_and_fragment_names(context, cached_fields_and_fragment_names, parent_type, selection_set)
    collect_conflicts_within(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, field_map)
    if fragment_names:
        for i, fragment_name in enumerate(fragment_names):
            collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, False, field_map, fragment_name)
            for other_fragment_name in fragment_names[i + 1:]:
                collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, False, fragment_name, other_fragment_name)
    return conflicts