import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def compare_codon(codon1, codon2, codon_table, weight=1):
    """Compare two codon accounting for different pathways."""
    sd = nd = 0
    if len(set(map(codon_table.forward_table.get, [codon1, codon2]))) == 1:
        sd += weight
    else:
        nd += weight
    return (sd, nd)