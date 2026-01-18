from typing import Any, Dict, Optional, Tuple
from ..types import FloatsXd
from ..util import get_array_module
def set_grad(self, model_id: int, name: str, value: FloatsXd) -> None:
    if self.proxy is not None:
        self.proxy.set_grad(model_id, name, value)
    else:
        self._grads[model_id, name] = value