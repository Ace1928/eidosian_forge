import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def get_aa_from_codonre(re_aa):
    aas = []
    m = 0
    for i in re_aa:
        if i == '[':
            m = -1
            aas.append('')
        elif i == ']':
            m = 0
            continue
        elif m == -1:
            aas[-1] = aas[-1] + i
        elif m == 0:
            aas.append(i)
    return aas