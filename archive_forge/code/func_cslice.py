from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def cslice(p):
    aa_slice = aa_index[p]
    codon_slice = ''
    for i in aa_slice:
        codon_slice += self[i * 3:i * 3 + 3]
    return str(codon_slice)