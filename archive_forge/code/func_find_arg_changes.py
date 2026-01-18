from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_arg_changes(old_type: Union[GraphQLObjectType, GraphQLInterfaceType], field_name: str, old_field: GraphQLField, new_field: GraphQLField) -> List[Change]:
    schema_changes: List[Change] = []
    args_diff = dict_diff(old_field.args, new_field.args)
    for arg_name in args_diff.removed:
        schema_changes.append(BreakingChange(BreakingChangeType.ARG_REMOVED, f'{old_type.name}.{field_name} arg {arg_name} was removed.'))
    for arg_name, (old_arg, new_arg) in args_diff.persisted.items():
        is_safe = is_change_safe_for_input_object_field_or_field_arg(old_arg.type, new_arg.type)
        if not is_safe:
            schema_changes.append(BreakingChange(BreakingChangeType.ARG_CHANGED_KIND, f'{old_type.name}.{field_name} arg {arg_name} has changed type from {old_arg.type} to {new_arg.type}.'))
        elif old_arg.default_value is not Undefined:
            if new_arg.default_value is Undefined:
                schema_changes.append(DangerousChange(DangerousChangeType.ARG_DEFAULT_VALUE_CHANGE, f'{old_type.name}.{field_name} arg {arg_name} defaultValue was removed.'))
            else:
                old_value_str = stringify_value(old_arg.default_value, old_arg.type)
                new_value_str = stringify_value(new_arg.default_value, new_arg.type)
                if old_value_str != new_value_str:
                    schema_changes.append(DangerousChange(DangerousChangeType.ARG_DEFAULT_VALUE_CHANGE, f'{old_type.name}.{field_name} arg {arg_name} has changed defaultValue from {old_value_str} to {new_value_str}.'))
    for arg_name, new_arg in args_diff.added.items():
        if is_required_argument(new_arg):
            schema_changes.append(BreakingChange(BreakingChangeType.REQUIRED_ARG_ADDED, f'A required arg {arg_name} on {old_type.name}.{field_name} was added.'))
        else:
            schema_changes.append(DangerousChange(DangerousChangeType.OPTIONAL_ARG_ADDED, f'An optional arg {arg_name} on {old_type.name}.{field_name} was added.'))
    return schema_changes