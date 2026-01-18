from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
@classmethod
def from_seq(cls, seq, rf_table=None):
    """Get codon sequence from sequence data."""
    if rf_table is None:
        return cls(str(seq))
    else:
        return cls(str(seq), rf_table=rf_table)