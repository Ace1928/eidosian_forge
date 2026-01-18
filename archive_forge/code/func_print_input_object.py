from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_input_object(type_: GraphQLInputObjectType) -> str:
    fields = [print_description(field, '  ', not i) + '  ' + print_input_value(name, field) for i, (name, field) in enumerate(type_.fields.items())]
    return print_description(type_) + f'input {type_.name}' + print_block(fields)