import random
import sys
from . import Nodes
def merge_with_support(self, bstrees=None, constree=None, threshold=0.5, outgroup=None):
    """Merge clade support (from consensus or list of bootstrap-trees) with phylogeny.

        tree=merge_bootstrap(phylo,bs_tree=<list_of_trees>)
        or
        tree=merge_bootstrap(phylo,consree=consensus_tree with clade support)
        """
    if bstrees and constree:
        raise TreeError('Specify either list of bootstrap trees or consensus tree, not both')
    if not (bstrees or constree):
        raise TreeError('Specify either list of bootstrap trees or consensus tree.')
    if outgroup is None:
        try:
            succnodes = self.node(self.root).succ
            smallest = min(((len(self.get_taxa(n)), n) for n in succnodes))
            outgroup = self.get_taxa(smallest[1])
        except Exception:
            raise TreeError('Error determining outgroup.') from None
    else:
        self.root_with_outgroup(outgroup)
    if bstrees:
        constree = consensus(bstrees, threshold=threshold, outgroup=outgroup)
    else:
        if not constree.has_support():
            constree.branchlength2support()
        constree.root_with_outgroup(outgroup)
    for pnode in self._walk():
        cnode = constree.is_monophyletic(self.get_taxa(pnode))
        if cnode > -1:
            self.node(pnode).data.support = constree.node(cnode).data.support