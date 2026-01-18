import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
@classmethod
def from_seqfeature(cls, feat):
    """Create ProteinDomain object from SeqFeature."""
    return ProteinDomain(feat.id, feat.location.start, feat.location.end, confidence=feat.qualifiers.get('confidence'))