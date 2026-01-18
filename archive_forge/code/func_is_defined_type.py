from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def is_defined_type(type_: GraphQLNamedType) -> bool:
    return not is_specified_scalar_type(type_) and (not is_introspection_type(type_))