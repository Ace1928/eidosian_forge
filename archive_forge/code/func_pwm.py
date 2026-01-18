from urllib.parse import urlencode
from urllib.request import urlopen, Request
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.Align import Alignment
from Bio.Seq import reverse_complement
@property
def pwm(self):
    """Calculate and return the position weight matrix for this motif."""
    return self.counts.normalize(self._pseudocounts)