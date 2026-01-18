from typing import List, Optional
from torchgen.api import dispatcher
from torchgen.api.types import (
from torchgen.model import (
def inner_arguments(func: FunctionSchema, is_reverse: bool) -> List[Binding]:
    args = func.arguments.flat_all
    assert args[0].type == BaseType(BaseTy.Tensor)
    non_self_args = args[1:]
    non_self_bindings = [dispatcher.argument(a) for a in non_self_args]
    if not is_reverse:
        return [base_binding] + non_self_bindings
    else:
        index_binding = inner_call_index(func)
        if index_binding is not None:
            return [base_binding, mutated_view_binding, reapply_views_binding, index_binding] + non_self_bindings
        else:
            return [base_binding, mutated_view_binding, reapply_views_binding] + non_self_bindings