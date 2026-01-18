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
def hashedDomainsByEv(self, id):
    """Get domains clustered by evalue in a dict."""
    if id not in self.EvDatahash:
        self.EvDatahash[id] = {}
        for d in self.domainsClusteredByEv(id):
            self.EvDatahash[id][d] = 1
    return self.EvDatahash[id]