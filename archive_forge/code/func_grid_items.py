import numpy as np
import param
from . import traversal
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .ndmapping import NdMapping, UniformNdMapping
def grid_items(self):
    return {tuple(np.unravel_index(idx, self.shape)): (path, item) for idx, (path, item) in enumerate(self.items())}