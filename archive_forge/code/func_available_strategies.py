from typing import Any, Callable, Dict, List, Optional
from typing_extensions import override
def available_strategies(self) -> List:
    """Returns a list of registered strategies."""
    return list(self.keys())