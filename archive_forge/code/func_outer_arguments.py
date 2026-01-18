from typing import List, Optional
from torchgen.api import dispatcher
from torchgen.api.types import (
from torchgen.model import (
def outer_arguments(*, is_reverse: bool) -> List[Binding]:
    if is_reverse:
        return [base_binding, mutated_view_binding, mutated_view_idx_binding]
    else:
        return [base_binding, mutated_view_idx_binding]