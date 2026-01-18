from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def get_codon_num(self):
    """Return the number of codons in the CodonSeq."""
    return len(self.rf_table)