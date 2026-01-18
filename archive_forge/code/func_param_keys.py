from typing import Any, Dict, Optional, Tuple
from ..types import FloatsXd
from ..util import get_array_module
@property
def param_keys(self) -> Tuple[KeyT, ...]:
    """Get the names of registered parameter (including unset)."""
    return tuple(self._params.keys())