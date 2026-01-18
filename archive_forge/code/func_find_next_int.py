import copy
from collections.abc import Mapping, Iterable
from Bio import BiopythonWarning
from Bio import BiopythonExperimentalWarning
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.codonalign.codonseq import CodonSeq
from Bio.codonalign.codonalignment import CodonAlignment, mktest
import warnings
def find_next_int(k, lst):
    idx = lst.index(k)
    p = 0
    while True:
        if isinstance(lst[idx + p], int):
            return (lst[idx + p], p)
        p += 1