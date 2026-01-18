import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def bootstrap_consensus(alignment, times, tree_constructor, consensus):
    """Consensus tree of a series of bootstrap trees for a multiple sequence alignment.

    :Parameters:
        alignment : Alignment or MultipleSeqAlignment object
            Multiple sequence alignment to generate replicates.
        times : int
            Number of bootstrap times.
        tree_constructor : TreeConstructor
            Tree constructor to be used to build trees.
        consensus : function
            Consensus method in this module: ``strict_consensus``,
            ``majority_consensus``, ``adam_consensus``.

    """
    trees = bootstrap_trees(alignment, times, tree_constructor)
    tree = consensus(trees)
    return tree