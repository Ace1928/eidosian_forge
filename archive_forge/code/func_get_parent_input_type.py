from typing import Any, Callable, List, Optional, Union, cast
from ..language import (
from ..pyutils import Undefined
from ..type import (
from .type_from_ast import type_from_ast
def get_parent_input_type(self) -> Optional[GraphQLInputType]:
    if len(self._input_type_stack) > 1:
        return self._input_type_stack[-2]
    return None