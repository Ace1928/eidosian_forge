import warnings
from typing import Any, Dict, Optional, Tuple
import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from torch.types import _size
@property
def arg_constraints(self) -> Dict[str, constraints.Constraint]:
    """
        Returns a dictionary from argument names to
        :class:`~torch.distributions.constraints.Constraint` objects that
        should be satisfied by each argument of this distribution. Args that
        are not tensors need not appear in this dict.
        """
    raise NotImplementedError