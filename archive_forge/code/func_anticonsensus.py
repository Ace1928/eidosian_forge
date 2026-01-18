from urllib.parse import urlencode
from urllib.request import urlopen, Request
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.Align import Alignment
from Bio.Seq import reverse_complement
@property
def anticonsensus(self):
    """Return the least probable pattern to be generated from this motif."""
    return self.counts.anticonsensus