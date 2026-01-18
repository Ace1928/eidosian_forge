import random
import sys
from . import Nodes
def newickize(node, ladderize=None):
    """Convert a node tree to a newick tree recursively."""
    if not self.node(node).succ:
        return self.node(node).data.taxon + make_info_string(self.node(node).data, terminal=True)
    else:
        succnodes = ladderize_nodes(self.node(node).succ, ladderize=ladderize)
        subtrees = [newickize(sn, ladderize=ladderize) for sn in succnodes]
        return f'({','.join(subtrees)}){make_info_string(self.node(node).data)}'