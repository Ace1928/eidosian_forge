import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def bootstrap_trees(alignment, times, tree_constructor):
    """Generate bootstrap replicate trees from a multiple sequence alignment.

    :Parameters:
        alignment : Alignment or MultipleSeqAlignment object
            multiple sequence alignment to generate replicates.
        times : int
            number of bootstrap times.
        tree_constructor : TreeConstructor
            tree constructor to be used to build trees.

    """
    if isinstance(alignment, MultipleSeqAlignment):
        length = len(alignment[0])
        for i in range(times):
            bootstrapped_alignment = None
            for j in range(length):
                col = random.randint(0, length - 1)
                if bootstrapped_alignment is None:
                    bootstrapped_alignment = alignment[:, col:col + 1]
                else:
                    bootstrapped_alignment += alignment[:, col:col + 1]
            tree = tree_constructor.build_tree(bootstrapped_alignment)
            yield tree
    else:
        n, m = alignment.shape
        for i in range(times):
            cols = [random.randint(0, m - 1) for j in range(m)]
            tree = tree_constructor.build_tree(alignment[:, cols])
            yield tree