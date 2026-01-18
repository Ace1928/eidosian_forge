from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def full_translate(self, codon_table=None, stop_symbol='*'):
    """Apply full translation with gaps considered."""
    if codon_table is None:
        codon_table = CodonTable.generic_by_id[1]
    full_rf_table = self.get_full_rf_table()
    return self.translate(codon_table=codon_table, stop_symbol=stop_symbol, rf_table=full_rf_table, ungap_seq=False)