from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_specified_by_url(scalar: GraphQLScalarType) -> str:
    if scalar.specified_by_url is None:
        return ''
    ast_value = print_ast(StringValueNode(value=scalar.specified_by_url))
    return f' @specifiedBy(url: {ast_value})'