from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def fix_single(tree):
    if isinstance(tree, PX.Phylogeny):
        return tree
    if isinstance(tree, PX.Clade):
        return tree.to_phylogeny()
    if isinstance(tree, PX.BaseTree.Tree):
        return PX.Phylogeny.from_tree(tree)
    if isinstance(tree, PX.BaseTree.Clade):
        return PX.Phylogeny.from_tree(PX.BaseTree.Tree(root=tree))
    else:
        raise ValueError('iterable must contain Tree or Clade types')