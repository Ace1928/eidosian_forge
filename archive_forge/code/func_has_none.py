import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
@property
def has_none(self):
    """
        Indicate if this BlockVector has any none entries.
        """
    return len(self._undefined_brows) != 0