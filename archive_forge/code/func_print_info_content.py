import math
import sys
import warnings
from collections import Counter
from Bio import BiopythonDeprecationWarning
from Bio.Seq import Seq
def print_info_content(summary_info, fout=None, rep_record=0):
    """3 column output: position, aa in representative sequence, ic_vector value."""
    warnings.warn('The `print_info_content` function is deprecated and will be removed in a future release of Biopython.', BiopythonDeprecationWarning)
    fout = fout or sys.stdout
    if not summary_info.ic_vector:
        summary_info.information_content()
    rep_sequence = summary_info.alignment[rep_record]
    for pos, (aa, ic) in enumerate(zip(rep_sequence, summary_info.ic_vector)):
        fout.write('%d %s %.3f\n' % (pos, aa, ic))