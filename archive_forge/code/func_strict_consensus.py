import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def strict_consensus(trees):
    """Search strict consensus tree from multiple trees.

    :Parameters:
        trees : iterable
            iterable of trees to produce consensus tree.

    """
    trees_iter = iter(trees)
    first_tree = next(trees_iter)
    terms = first_tree.get_terminals()
    bitstr_counts, tree_count = _count_clades(itertools.chain([first_tree], trees_iter))
    strict_bitstrs = [bitstr for bitstr, t in bitstr_counts.items() if t[0] == tree_count]
    strict_bitstrs.sort(key=lambda bitstr: bitstr.count('1'), reverse=True)
    root = BaseTree.Clade()
    if strict_bitstrs[0].count('1') == len(terms):
        root.clades.extend(terms)
    else:
        raise ValueError('Taxons in provided trees should be consistent')
    bitstr_clades = {strict_bitstrs[0]: root}
    for bitstr in strict_bitstrs[1:]:
        clade_terms = [terms[i] for i in bitstr.index_one()]
        clade = BaseTree.Clade()
        clade.clades.extend(clade_terms)
        for bs, c in bitstr_clades.items():
            if bs.contains(bitstr):
                del bitstr_clades[bs]
                new_childs = [child for child in c.clades if child not in clade_terms]
                c.clades = new_childs
                c.clades.append(clade)
                bs = bs ^ bitstr
                bitstr_clades[bs] = c
                break
        bitstr_clades[bitstr] = clade
    return BaseTree.Tree(root=root)