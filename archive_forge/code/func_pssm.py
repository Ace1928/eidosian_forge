from urllib.parse import urlencode
from urllib.request import urlopen, Request
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.Align import Alignment
from Bio.Seq import reverse_complement
@property
def pssm(self):
    """Calculate and return the position specific scoring matrix for this motif."""
    return self.pwm.log_odds(self._background)