import warnings
from typing import Any, Dict, Optional, Tuple
import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from torch.types import _size
@staticmethod
def set_default_validate_args(value: bool) -> None:
    """
        Sets whether validation is enabled or disabled.

        The default behavior mimics Python's ``assert`` statement: validation
        is on by default, but is disabled if Python is run in optimized mode
        (via ``python -O``). Validation may be expensive, so you may want to
        disable it once a model is working.

        Args:
            value (bool): Whether to enable validation.
        """
    if value not in [True, False]:
        raise ValueError
    Distribution._validate_args = value