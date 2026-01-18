import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
def nj(self, distance_matrix):
    """Construct and return a Neighbor Joining tree.

        :Parameters:
            distance_matrix : DistanceMatrix
                The distance matrix for tree construction.

        """
    if not isinstance(distance_matrix, DistanceMatrix):
        raise TypeError('Must provide a DistanceMatrix object.')
    dm = copy.deepcopy(distance_matrix)
    clades = [BaseTree.Clade(None, name) for name in dm.names]
    node_dist = [0] * len(dm)
    min_i = 0
    min_j = 0
    inner_count = 0
    if len(dm) == 1:
        root = clades[0]
        return BaseTree.Tree(root, rooted=False)
    elif len(dm) == 2:
        min_i = 1
        min_j = 0
        clade1 = clades[min_i]
        clade2 = clades[min_j]
        clade1.branch_length = dm[min_i, min_j] / 2.0
        clade2.branch_length = dm[min_i, min_j] - clade1.branch_length
        inner_clade = BaseTree.Clade(None, 'Inner')
        inner_clade.clades.append(clade1)
        inner_clade.clades.append(clade2)
        clades[0] = inner_clade
        root = clades[0]
        return BaseTree.Tree(root, rooted=False)
    while len(dm) > 2:
        for i in range(0, len(dm)):
            node_dist[i] = 0
            for j in range(0, len(dm)):
                node_dist[i] += dm[i, j]
            node_dist[i] = node_dist[i] / (len(dm) - 2)
        min_dist = dm[1, 0] - node_dist[1] - node_dist[0]
        min_i = 0
        min_j = 1
        for i in range(1, len(dm)):
            for j in range(0, i):
                temp = dm[i, j] - node_dist[i] - node_dist[j]
                if min_dist > temp:
                    min_dist = temp
                    min_i = i
                    min_j = j
        clade1 = clades[min_i]
        clade2 = clades[min_j]
        inner_count += 1
        inner_clade = BaseTree.Clade(None, 'Inner' + str(inner_count))
        inner_clade.clades.append(clade1)
        inner_clade.clades.append(clade2)
        clade1.branch_length = (dm[min_i, min_j] + node_dist[min_i] - node_dist[min_j]) / 2.0
        clade2.branch_length = dm[min_i, min_j] - clade1.branch_length
        clades[min_j] = inner_clade
        del clades[min_i]
        for k in range(0, len(dm)):
            if k != min_i and k != min_j:
                dm[min_j, k] = (dm[min_i, k] + dm[min_j, k] - dm[min_i, min_j]) / 2.0
        dm.names[min_j] = 'Inner' + str(inner_count)
        del dm[min_i]
    root = None
    if clades[0] == inner_clade:
        clades[0].branch_length = 0
        clades[1].branch_length = dm[1, 0]
        clades[0].clades.append(clades[1])
        root = clades[0]
    else:
        clades[0].branch_length = dm[1, 0]
        clades[1].branch_length = 0
        clades[1].clades.append(clades[0])
        root = clades[1]
    return BaseTree.Tree(root, rooted=False)