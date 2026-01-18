from enum import Enum
from typing import Any, Collection, Dict, List, NamedTuple, Union, cast
from ..language import print_ast
from ..pyutils import inspect, Undefined
from ..type import (
from ..utilities.sort_value_node import sort_value_node
from .ast_from_value import ast_from_value
def find_breaking_changes(old_schema: GraphQLSchema, new_schema: GraphQLSchema) -> List[BreakingChange]:
    """Find breaking changes.

    Given two schemas, returns a list containing descriptions of all the types of
    breaking changes covered by the other functions down below.
    """
    return [change for change in find_schema_changes(old_schema, new_schema) if isinstance(change.type, BreakingChangeType)]