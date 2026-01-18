import random
import numpy as np
from ...block import Block, HybridBlock
from ...nn import Sequential, HybridSequential
from .... import image
from ....base import numeric_types
from ....util import is_np_array
def hybrid_forward(self, F, x):
    if is_np_array():
        F = F.npx
    return F.image.random_lighting(x, self._alpha)