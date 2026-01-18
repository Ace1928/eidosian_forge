import copy
from typing import Any, Dict
import stanio
@property
def method_vars(self) -> Dict[str, stanio.Variable]:
    """
        Method variable names always end in `__`, e.g. `lp__`.
        """
    return self._method_vars