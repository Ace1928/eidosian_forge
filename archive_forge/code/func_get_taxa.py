import random
import sys
from . import Nodes
def get_taxa(self, node_id=None):
    """Return a list of all otus downwards from a node.

        nodes = get_taxa(self,node_id=None)
        """
    if node_id is None:
        node_id = self.root
    if node_id not in self.chain:
        raise TreeError('Unknown node_id: %d.' % node_id)
    if self.chain[node_id].succ == []:
        if self.chain[node_id].data:
            return [self.chain[node_id].data.taxon]
        else:
            return None
    else:
        list = []
        for succ in self.chain[node_id].succ:
            list.extend(self.get_taxa(succ))
        return list