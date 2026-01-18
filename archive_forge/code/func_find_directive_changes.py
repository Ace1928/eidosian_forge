from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_directive_changes(old_schema: GraphQLSchema, new_schema: GraphQLSchema) -> List[Change]:
    schema_changes: List[Change] = []
    directives_diff = list_diff(old_schema.directives, new_schema.directives)
    for directive in directives_diff.removed:
        schema_changes.append(BreakingChange(BreakingChangeType.DIRECTIVE_REMOVED, f'{directive.name} was removed.'))
    for old_directive, new_directive in directives_diff.persisted:
        args_diff = dict_diff(old_directive.args, new_directive.args)
        for arg_name, new_arg in args_diff.added.items():
            if is_required_argument(new_arg):
                schema_changes.append(BreakingChange(BreakingChangeType.REQUIRED_DIRECTIVE_ARG_ADDED, f'A required arg {arg_name} on directive {old_directive.name} was added.'))
        for arg_name in args_diff.removed:
            schema_changes.append(BreakingChange(BreakingChangeType.DIRECTIVE_ARG_REMOVED, f'{arg_name} was removed from {new_directive.name}.'))
        if old_directive.is_repeatable and (not new_directive.is_repeatable):
            schema_changes.append(BreakingChange(BreakingChangeType.DIRECTIVE_REPEATABLE_REMOVED, f'Repeatable flag was removed from {old_directive.name}.'))
        for location in old_directive.locations:
            if location not in new_directive.locations:
                schema_changes.append(BreakingChange(BreakingChangeType.DIRECTIVE_LOCATION_REMOVED, f'{location.name} was removed from {new_directive.name}.'))
    return schema_changes