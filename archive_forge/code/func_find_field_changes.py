from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_field_changes(old_type: Union[GraphQLObjectType, GraphQLInterfaceType], new_type: Union[GraphQLObjectType, GraphQLInterfaceType]) -> List[Change]:
    schema_changes: List[Change] = []
    fields_diff = dict_diff(old_type.fields, new_type.fields)
    for field_name in fields_diff.removed:
        schema_changes.append(BreakingChange(BreakingChangeType.FIELD_REMOVED, f'{old_type.name}.{field_name} was removed.'))
    for field_name, (old_field, new_field) in fields_diff.persisted.items():
        schema_changes.extend(find_arg_changes(old_type, field_name, old_field, new_field))
        is_safe = is_change_safe_for_object_or_interface_field(old_field.type, new_field.type)
        if not is_safe:
            schema_changes.append(BreakingChange(BreakingChangeType.FIELD_CHANGED_KIND, f'{old_type.name}.{field_name} changed type from {old_field.type} to {new_field.type}.'))
    return schema_changes