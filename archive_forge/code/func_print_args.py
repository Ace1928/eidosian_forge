from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_args(args: Dict[str, GraphQLArgument], indentation: str='') -> str:
    if not args:
        return ''
    if not any((arg.description for arg in args.values())):
        return '(' + ', '.join((print_input_value(name, arg) for name, arg in args.items())) + ')'
    return '(\n' + '\n'.join((print_description(arg, f'  {indentation}', not i) + f'  {indentation}' + print_input_value(name, arg) for i, (name, arg) in enumerate(args.items()))) + f'\n{indentation})'