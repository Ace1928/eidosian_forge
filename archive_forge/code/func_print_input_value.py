from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_input_value(name: str, arg: GraphQLArgument) -> str:
    default_ast = ast_from_value(arg.default_value, arg.type)
    arg_decl = f'{name}: {arg.type}'
    if default_ast:
        arg_decl += f' = {print_ast(default_ast)}'
    return arg_decl + print_deprecated(arg.deprecation_reason)