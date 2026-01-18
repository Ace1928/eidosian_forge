from typing import Any, Dict, List, Set, Union, cast
from ..language import (
from ..type import (
from ..utilities.type_from_ast import type_from_ast
from .values import get_directive_values
def collect_sub_fields(schema: GraphQLSchema, fragments: Dict[str, FragmentDefinitionNode], variable_values: Dict[str, Any], return_type: GraphQLObjectType, field_nodes: List[FieldNode]) -> Dict[str, List[FieldNode]]:
    """Collect sub fields.

    Given a list of field nodes, collects all the subfields of the passed in fields,
    and returns them at the end.

    collect_sub_fields requires the "return type" of an object. For a field that
    returns an Interface or Union type, the "return type" will be the actual
    object type returned by that field.

    For internal use only.
    """
    sub_field_nodes: Dict[str, List[FieldNode]] = {}
    visited_fragment_names: Set[str] = set()
    for node in field_nodes:
        if node.selection_set:
            collect_fields_impl(schema, fragments, variable_values, return_type, node.selection_set, sub_field_nodes, visited_fragment_names)
    return sub_field_nodes