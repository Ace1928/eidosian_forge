from typing import List, Optional
from torchgen.api import dispatcher
from torchgen.api.types import (
from torchgen.model import (
def capture_arguments(func: FunctionSchema, *, is_reverse: bool) -> List[Binding]:
    args = func.arguments.flat_all
    assert args[0].type == BaseType(BaseTy.Tensor)
    non_self_args = args[1:]
    non_self_value_bindings = [dispatcher.argument(a, remove_non_owning_ref_types=True) for a in non_self_args]
    all_bindings = [reapply_views_binding] + non_self_value_bindings
    return all_bindings