from typing import Any, Callable, Collection, Dict, List, Optional, Union, cast
from ..error import GraphQLError
from ..language import (
from ..pyutils import inspect, print_path_list, Undefined
from ..type import (
from ..utilities.coerce_input_value import coerce_input_value
from ..utilities.type_from_ast import type_from_ast
from ..utilities.value_from_ast import value_from_ast
def on_input_value_error(path: List[Union[str, int]], invalid_value: Any, error: GraphQLError) -> None:
    invalid_str = inspect(invalid_value)
    prefix = f"Variable '${var_name}' got invalid value {invalid_str}"
    if path:
        prefix += f" at '{var_name}{print_path_list(path)}'"
    on_error(GraphQLError(prefix + '; ' + error.message, var_def_node, original_error=error.original_error))