from math import sqrt, erfc
import warnings
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio import BiopythonWarning
from Bio.codonalign.codonseq import _get_codon_list, CodonSeq, cal_dn_ds
@classmethod
def from_msa(cls, align):
    """Convert a MultipleSeqAlignment to CodonAlignment.

        Function to convert a MultipleSeqAlignment to CodonAlignment.
        It is the user's responsibility to ensure all the requirement
        needed by CodonAlignment is met.
        """
    rec = [SeqRecord(CodonSeq(str(i.seq)), id=i.id) for i in align._records]
    return cls(rec)