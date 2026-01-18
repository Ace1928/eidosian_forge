from typing import Any, Dict, Optional, Tuple
from ..types import FloatsXd
from ..util import get_array_module
def has_grad(self, model_id: int, name: str) -> bool:
    return (model_id, name) in self._grads