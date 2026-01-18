from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def collect_conflicts_between_fragments(context: ValidationContext, conflicts: List[Conflict], cached_fields_and_fragment_names: Dict, compared_fragment_pairs: 'PairSet', are_mutually_exclusive: bool, fragment_name1: str, fragment_name2: str) -> None:
    """Collect conflicts between fragments.

    Collect all conflicts found between two fragments, including via spreading in any
    nested fragments.
    """
    if fragment_name1 == fragment_name2:
        return
    if compared_fragment_pairs.has(fragment_name1, fragment_name2, are_mutually_exclusive):
        return
    compared_fragment_pairs.add(fragment_name1, fragment_name2, are_mutually_exclusive)
    fragment1 = context.get_fragment(fragment_name1)
    fragment2 = context.get_fragment(fragment_name2)
    if not fragment1 or not fragment2:
        return None
    field_map1, referenced_fragment_names1 = get_referenced_fields_and_fragment_names(context, cached_fields_and_fragment_names, fragment1)
    field_map2, referenced_fragment_names2 = get_referenced_fields_and_fragment_names(context, cached_fields_and_fragment_names, fragment2)
    collect_conflicts_between(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, field_map1, field_map2)
    for referenced_fragment_name2 in referenced_fragment_names2:
        collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, fragment_name1, referenced_fragment_name2)
    for referenced_fragment_name1 in referenced_fragment_names1:
        collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragment_pairs, are_mutually_exclusive, referenced_fragment_name1, fragment_name2)