from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from ...error import GraphQLError
from ...language import (
from ...type import (
from ...utilities import type_from_ast
from ...utilities.sort_value_node import sort_value_node
from . import ValidationContext, ValidationRule
def stringify_arguments(field_node: FieldNode) -> str:
    input_object_with_args = ObjectValueNode(fields=tuple((ObjectFieldNode(name=arg_node.name, value=arg_node.value) for arg_node in field_node.arguments)))
    return print_ast(sort_value_node(input_object_with_args))