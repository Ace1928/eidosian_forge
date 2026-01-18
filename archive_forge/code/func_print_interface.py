from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_interface(type_: GraphQLInterfaceType) -> str:
    return print_description(type_) + f'interface {type_.name}' + print_implemented_interfaces(type_) + print_fields(type_)