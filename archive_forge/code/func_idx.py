import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
@property
def idx(self):
    return self._node.indices