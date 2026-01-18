from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def collect_conflicts_between(context: ValidationContext, conflicts: List[Conflict], cached_fields_and_fragment_names: Dict, compared_fragment_pairs: 'PairSet', parent_fields_are_mutually_exclusive: bool, field_map1: NodeAndDefCollection, field_map2: NodeAndDefCollection) -> None:
    """Collect all Conflicts between two collections of fields.

    This is similar to, but different from the :func:`~.collect_conflicts_within`
    function above. This check assumes that :func:`~.collect_conflicts_within` has
    already been called on each provided collection of fields. This is true because
    this validator traverses each individual selection set.
    """
    for response_name, fields1 in field_map1.items():
        fields2 = field_map2.get(response_name)
        if fields2:
            for field1 in fields1:
                for field2 in fields2:
                    conflict = find_conflict(context, cached_fields_and_fragment_names, compared_fragment_pairs, parent_fields_are_mutually_exclusive, response_name, field1, field2)
                    if conflict:
                        conflicts.append(conflict)