import random
import sys
from . import Nodes
def search_taxon(self, taxon):
    """Return the first matching taxon in self.data.taxon. Not restricted to terminal nodes.

        node_id = search_taxon(self,taxon)

        """
    for id, node in self.chain.items():
        if node.data.taxon == taxon:
            return id
    return None