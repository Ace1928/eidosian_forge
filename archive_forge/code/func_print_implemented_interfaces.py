from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_implemented_interfaces(type_: Union[GraphQLObjectType, GraphQLInterfaceType]) -> str:
    interfaces = type_.interfaces
    return ' implements ' + ' & '.join((i.name for i in interfaces)) if interfaces else ''