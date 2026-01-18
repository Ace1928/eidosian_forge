from urllib.parse import urlencode
from urllib.request import urlopen, Request
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.Align import Alignment
from Bio.Seq import reverse_complement
@property
def degenerate_consensus(self):
    """Return the degenerate consensus sequence.

        Following the rules adapted from
        D. R. Cavener: "Comparison of the consensus sequence flanking
        translational start sites in Drosophila and vertebrates."
        Nucleic Acids Research 15(4): 1353-1361. (1987).

        The same rules are used by TRANSFAC.
        """
    return self.counts.degenerate_consensus