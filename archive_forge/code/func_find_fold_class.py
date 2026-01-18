from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def find_fold_class(codon, forward_table):
    base = {'A', 'T', 'C', 'G'}
    fold = ''
    codon_base_lst = list(codon)
    for i, b in enumerate(codon_base_lst):
        other_base = base - set(b)
        aa = []
        for j in other_base:
            codon_base_lst[i] = j
            try:
                aa.append(forward_table[''.join(codon_base_lst)])
            except KeyError:
                aa.append('stop')
        if aa.count(forward_table[codon]) == 0:
            fold += '0'
        elif aa.count(forward_table[codon]) in (1, 2):
            fold += '2'
        elif aa.count(forward_table[codon]) == 3:
            fold += '4'
        else:
            raise RuntimeError('Unknown Error, cannot assign the position to a fold')
        codon_base_lst[i] = b
    return fold