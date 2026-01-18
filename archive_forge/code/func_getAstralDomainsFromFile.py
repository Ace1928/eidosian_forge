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
def getAstralDomainsFromFile(self, filename=None, file_handle=None):
    """Get the scop domains from a file containing a list of sids."""
    if file_handle is None and filename is None:
        raise RuntimeError('You must provide a filename or handle')
    if not file_handle:
        file_handle = open(filename)
    doms = []
    while True:
        line = file_handle.readline()
        if not line:
            break
        line = line.rstrip()
        doms.append(line)
    if filename:
        file_handle.close()
    doms = [a for a in doms if a[0] == 'd']
    doms = [self.scop.getDomainBySid(x) for x in doms]
    return doms