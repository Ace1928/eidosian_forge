import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def calculate_dn_ds_matrix(alignment, method='NG86', codon_table=None):
    """Calculate dN and dS pairwise for the multiple alignment, and return as matrices.

    Argument:
     - method       - Available methods include NG86, LWL85, YN00 and ML.
     - codon_table  - Codon table to use for forward translation.

    """
    from Bio.Phylo.TreeConstruction import DistanceMatrix
    if codon_table is None:
        codon_table = CodonTable.generic_by_id[1]
    sequences = alignment.sequences
    coordinates = alignment.coordinates
    names = [record.id for record in sequences]
    size = len(names)
    dn_matrix = []
    ds_matrix = []
    for i in range(size):
        dn_matrix.append([])
        ds_matrix.append([])
        for j in range(i):
            pairwise_sequences = [sequences[i], sequences[j]]
            pairwise_coordinates = coordinates[(i, j), :]
            pairwise_alignment = Alignment(pairwise_sequences, pairwise_coordinates)
            dn, ds = calculate_dn_ds(pairwise_alignment, method=method, codon_table=codon_table)
            dn_matrix[i].append(dn)
            ds_matrix[i].append(ds)
        dn_matrix[i].append(0.0)
        ds_matrix[i].append(0.0)
    dn_dm = DistanceMatrix(names, matrix=dn_matrix)
    ds_dm = DistanceMatrix(names, matrix=ds_matrix)
    return (dn_dm, ds_dm)