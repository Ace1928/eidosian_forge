import sys
from math import sqrt, erfc, log, floor
from heapq import heapify, heappop, heappush
from itertools import permutations
from collections import defaultdict, Counter
import numpy as np
from Bio.Align import Alignment
from Bio.Data import CodonTable
def calculate_dn_ds(alignment, method='NG86', codon_table=None, k=1, cfreq=None):
    """Calculate dN and dS of the given two sequences.

    Available methods:
        - NG86  - `Nei and Gojobori (1986)`_ (PMID 3444411).
        - LWL85 - `Li et al. (1985)`_ (PMID 3916709).
        - ML    - `Goldman and Yang (1994)`_ (PMID 7968486).
        - YN00  - `Yang and Nielsen (2000)`_ (PMID 10666704).

    .. _`Nei and Gojobori (1986)`: http://www.ncbi.nlm.nih.gov/pubmed/3444411
    .. _`Li et al. (1985)`: http://www.ncbi.nlm.nih.gov/pubmed/3916709
    .. _`Goldman and Yang (1994)`: http://mbe.oxfordjournals.org/content/11/5/725
    .. _`Yang and Nielsen (2000)`: https://doi.org/10.1093/oxfordjournals.molbev.a026236

    Arguments:
     - k  - transition/transversion rate ratio
     - cfreq - Current codon frequency vector can only be specified
       when you are using ML method. Possible ways of
       getting cfreq are: F1x4, F3x4 and F61.

    """
    if cfreq is None:
        cfreq = 'F3x4'
    elif cfreq is not None and method != 'ML':
        raise ValueError('cfreq can only be specified when you are using ML method')
    elif cfreq not in ('F1x4', 'F3x4', 'F61'):
        raise ValueError("cfreq must be 'F1x4', 'F3x4', or 'F61'")
    if codon_table is None:
        codon_table = CodonTable.generic_by_id[1]
    codons1 = []
    codons2 = []
    sequence1, sequence2 = alignment.sequences
    try:
        sequence1 = sequence1.seq
    except AttributeError:
        pass
    sequence1 = str(sequence1)
    try:
        sequence2 = sequence2.seq
    except AttributeError:
        pass
    sequence2 = str(sequence2)
    aligned1, aligned2 = alignment.aligned
    for block1, block2 in zip(aligned1, aligned2):
        start1, end1 = block1
        start2, end2 = block2
        codons1.extend((sequence1[i:i + 3] for i in range(start1, end1, 3)))
        codons2.extend((sequence2[i:i + 3] for i in range(start2, end2, 3)))
    bases = {'A', 'T', 'C', 'G'}
    for codon1 in codons1:
        if not all((nucleotide in bases for nucleotide in codon1)):
            raise ValueError(f'Unrecognized character in {codon1} in the target sequence (Codons consist of A, T, C or G)')
    for codon2 in codons2:
        if not all((nucleotide in bases for nucleotide in codon2)):
            raise ValueError(f'Unrecognized character in {codon2} in the query sequence (Codons consist of A, T, C or G)')
    if method == 'ML':
        return _ml(codons1, codons2, cfreq, codon_table)
    elif method == 'NG86':
        return _ng86(codons1, codons2, k, codon_table)
    elif method == 'LWL85':
        return _lwl85(codons1, codons2, codon_table)
    elif method == 'YN00':
        return _yn00(codons1, codons2, codon_table)
    else:
        raise ValueError(f"Unknown method '{method}'")