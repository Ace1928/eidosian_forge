from itertools import permutations
from math import log
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
def get_codon(self, index):
    """Get the index codon from the sequence."""
    if len({i % 3 for i in self.rf_table}) != 1:
        raise RuntimeError('frameshift detected. CodonSeq object is not able to deal with codon sequence with frameshift. Please use normal slice option.')
    if isinstance(index, int):
        if index != -1:
            return str(self[index * 3:(index + 1) * 3])
        else:
            return str(self[index * 3:])
    else:
        aa_index = range(len(self) // 3)

        def cslice(p):
            aa_slice = aa_index[p]
            codon_slice = ''
            for i in aa_slice:
                codon_slice += self[i * 3:i * 3 + 3]
            return str(codon_slice)
        codon_slice = cslice(index)
        return CodonSeq(codon_slice)