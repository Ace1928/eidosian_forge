import numpy as np
import param
from . import traversal
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .ndmapping import NdMapping, UniformNdMapping
@property
def ddims(self):
    return self.main.dimensions()