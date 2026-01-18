import operator
from ..dependencies import numpy as np
from .base_block import BaseBlockVector
def is_block_defined(self, ndx):
    return ndx not in self._undefined_brows