from typing import List, Optional
from torchgen.api import dispatcher
from torchgen.api.types import (
from torchgen.model import (
def inner_call_index(func: FunctionSchema) -> Optional[Binding]:
    if len(func.returns) > 1 or (len(func.returns) == 1 and func.returns[0].type.is_list_like()):
        return mutated_view_idx_binding
    return None