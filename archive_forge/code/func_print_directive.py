from typing import Any, Callable, Dict, List, Optional, Union, cast
from ..language import print_ast, StringValueNode
from ..language.block_string import is_printable_as_block_string
from ..pyutils import inspect
from ..type import (
from .ast_from_value import ast_from_value
def print_directive(directive: GraphQLDirective) -> str:
    return print_description(directive) + f'directive @{directive.name}' + print_args(directive.args) + (' repeatable' if directive.is_repeatable else '') + ' on ' + ' | '.join((location.name for location in directive.locations))