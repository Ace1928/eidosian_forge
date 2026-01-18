from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_filtered_schema(schema: GraphQLSchema, directive_filter: Callable[[GraphQLDirective], bool], type_filter: Callable[[GraphQLNamedType], bool]) -> str:
    directives = filter(directive_filter, schema.directives)
    types = filter(type_filter, schema.type_map.values())
    return '\n\n'.join((*filter(None, (print_schema_definition(schema),)), *map(print_directive, directives), *map(print_type, types)))