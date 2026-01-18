from typing import Any, Callable, Dict, List, Optional
from typing_extensions import override
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.registry import _register_classes
def available_accelerators(self) -> List[str]:
    """Returns a list of registered accelerators."""
    return list(self.keys())