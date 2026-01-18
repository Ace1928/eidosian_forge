import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def majority_consensus(trees, cutoff=0):
    """Search majority rule consensus tree from multiple trees.

    This is a extend majority rule method, which means the you can set any
    cutoff between 0 ~ 1 instead of 0.5. The default value of cutoff is 0 to
    create a relaxed binary consensus tree in any condition (as long as one of
    the provided trees is a binary tree). The branch length of each consensus
    clade in the result consensus tree is the average length of all counts for
    that clade.

    :Parameters:
        trees : iterable
            iterable of trees to produce consensus tree.

    """
    tree_iter = iter(trees)
    first_tree = next(tree_iter)
    terms = first_tree.get_terminals()
    bitstr_counts, tree_count = _count_clades(itertools.chain([first_tree], tree_iter))
    bitstrs = sorted(bitstr_counts.keys(), key=lambda bitstr: (bitstr_counts[bitstr][0], bitstr.count('1'), str(bitstr)), reverse=True)
    root = BaseTree.Clade()
    if bitstrs[0].count('1') == len(terms):
        root.clades.extend(terms)
    else:
        raise ValueError('Taxons in provided trees should be consistent')
    bitstr_clades = {bitstrs[0]: root}
    for bitstr in bitstrs[1:]:
        count_in_trees, branch_length_sum = bitstr_counts[bitstr]
        confidence = 100.0 * count_in_trees / tree_count
        if confidence < cutoff * 100.0:
            break
        clade_terms = [terms[i] for i in bitstr.index_one()]
        clade = BaseTree.Clade()
        clade.clades.extend(clade_terms)
        clade.confidence = confidence
        clade.branch_length = branch_length_sum / count_in_trees
        bsckeys = sorted(bitstr_clades, key=lambda bs: bs.count('1'), reverse=True)
        compatible = True
        parent_bitstr = None
        child_bitstrs = []
        for bs in bsckeys:
            if not bs.iscompatible(bitstr):
                compatible = False
                break
            if bs.contains(bitstr):
                parent_bitstr = bs
            if bitstr.contains(bs) and bs != bitstr and all((c.independent(bs) for c in child_bitstrs)):
                child_bitstrs.append(bs)
        if not compatible:
            continue
        if parent_bitstr:
            parent_clade = bitstr_clades.pop(parent_bitstr)
            parent_clade.clades = [c for c in parent_clade.clades if c not in clade_terms]
            parent_clade.clades.append(clade)
            bitstr_clades[parent_bitstr] = parent_clade
        if child_bitstrs:
            remove_list = []
            for c in child_bitstrs:
                remove_list.extend(c.index_one())
                child_clade = bitstr_clades[c]
                parent_clade.clades.remove(child_clade)
                clade.clades.append(child_clade)
            remove_terms = [terms[i] for i in remove_list]
            clade.clades = [c for c in clade.clades if c not in remove_terms]
        bitstr_clades[bitstr] = clade
        if len(bitstr_clades) == len(terms) - 1 or (len(bitstr_clades) == len(terms) - 2 and len(root.clades) == 3):
            break
    return BaseTree.Tree(root=root)