from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def collect_conflicts_between_fields_and_fragment(context: ValidationContext, conflicts: List[Conflict], cached_fields_and_fragment_names: Dict, compared_fragment_pairs: 'PairSet', are_mutually_exclusive: bool, field_map: NodeAndDefCollection, fragment_name: str) -> None:
    """Collect conflicts between fields and fragment.

    Collect all conflicts found between a set of fields and a fragment reference
    including via spreading in any nested fragments.
    """
    fragment = context.get_fragment(fragment_name)
    if not fragment:
        return None
    field_map2, referenced_fragment_names = get_referenced_fields_and_fragment_names(context, cached_fields_and_fragment_names, fragment)
    if field_map is field_map2:
        return
    collect_conflicts_between(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, field_map, field_map2)
    for referenced_fragment_name in referenced_fragment_names:
        if compared_fragment_pairs.has(referenced_fragment_name, fragment_name, are_mutually_exclusive):
            continue
        compared_fragment_pairs.add(referenced_fragment_name, fragment_name, are_mutually_exclusive)
        collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, field_map, referenced_fragment_name)