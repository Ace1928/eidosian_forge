from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def collect_conflicts_within(context: ValidationContext, conflicts: List[Conflict], cached_fields_and_fragment_names: Dict, compared_fragment_pairs: 'PairSet', field_map: NodeAndDefCollection) -> None:
    """Collect all Conflicts "within" one collection of fields."""
    for response_name, fields in field_map.items():
        if len(fields) > 1:
            for i, field in enumerate(fields):
                for other_field in fields[i + 1:]:
                    conflict = find_conflict(context, cached_fields_and_fragment_names, compared_fragment_pairs, False, response_name, field, other_field)
                    if conflict:
                        conflicts.append(conflict)