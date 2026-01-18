import random
import sys
from . import Nodes
def branchlength2support(self):
    """Move values stored in data.branchlength to data.support, and set branchlength to 0.0.

        This is necessary when support has been stored as branchlength (e.g. paup), and has thus
        been read in as branchlength.
        """
    for n in self.chain:
        self.node(n).data.support = self.node(n).data.branchlength
        self.node(n).data.branchlength = 0.0