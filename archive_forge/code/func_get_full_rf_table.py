from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def get_full_rf_table(self):
    """Return full rf_table of the CodonSeq records.

        A full rf_table is different from a normal rf_table in that
        it translate gaps in CodonSeq. It is helpful to construct
        alignment containing frameshift.
        """
    ungap_seq = str(self).replace('-', '')
    relative_pos = [self.rf_table[0]]
    for i in range(1, len(self.rf_table[1:]) + 1):
        relative_pos.append(self.rf_table[i] - self.rf_table[i - 1])
    full_rf_table = []
    codon_num = 0
    for i in range(0, len(self), 3):
        if self[i:i + 3] == self.gap_char * 3:
            full_rf_table.append(i + 0.0)
        elif relative_pos[codon_num] == 0:
            full_rf_table.append(i)
            codon_num += 1
        elif relative_pos[codon_num] in (-1, -2):
            gap_stat = 3 - self.count('-', i - 3, i)
            if gap_stat == 3:
                full_rf_table.append(i + relative_pos[codon_num])
            elif gap_stat == 2:
                full_rf_table.append(i + 1 + relative_pos[codon_num])
            elif gap_stat == 1:
                full_rf_table.append(i + 2 + relative_pos[codon_num])
            codon_num += 1
        elif relative_pos[codon_num] > 0:
            full_rf_table.append(i + 0.0)
        try:
            this_len = 3 - self.count('-', i, i + 3)
            relative_pos[codon_num] -= this_len
        except Exception:
            pass
    return full_rf_table