import os
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from . import Des
from . import Cla
from . import Hie
from . import Residues
from Bio import SeqIO
from Bio.Seq import Seq
def isDomainInEv(self, dom, id):
    """Return true if the domain is in the ASTRAL clusters for evalues."""
    return dom in self.hashedDomainsByEv(id)