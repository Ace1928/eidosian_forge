import random
import sys
from . import Nodes
def is_identical(self, tree2):
    """Compare tree and tree2 for identity.

        result = is_identical(self,tree2)
        """
    return self.set_subtree(self.root) == tree2.set_subtree(tree2.root)