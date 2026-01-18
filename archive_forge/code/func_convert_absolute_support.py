import random
import sys
from . import Nodes
def convert_absolute_support(self, nrep):
    """Convert absolute support (clade-count) to rel. frequencies.

        Some software (e.g. PHYLIP consense) just calculate how often clades appear, instead of
        calculating relative frequencies.
        """
    for n in self._walk():
        if self.node(n).data.support:
            self.node(n).data.support /= nrep