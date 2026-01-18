import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def is_aligned_seq(elem):
    if isinstance(elem, Sequence) and elem.mol_seq.is_aligned:
        return True
    return False