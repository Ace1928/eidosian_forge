import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
def get_coordinate(self) -> Optional[List[int]]:
    """
            Return the relative indices of this rank relative to all
            dimensions of the mesh. If this rank is not part of the mesh, return None.
            """
    return self._coordinate_on_dim if self._coordinate_on_dim else None