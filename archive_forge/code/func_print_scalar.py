from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_scalar(type_: GraphQLScalarType) -> str:
    return print_description(type_) + f'scalar {type_.name}' + print_specified_by_url(type_)