from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union, cast
from ..error import GraphQLError
from ..language import (
from ..type import (
from ..utilities import TypeInfo, TypeInfoVisitor
def enter_variable(self, node: VariableNode, *_args: Any) -> VisitorAction:
    type_info = self._type_info
    usage = VariableUsage(node, type_info.get_input_type(), type_info.get_default_value())
    self._append_usage(usage)
    return None