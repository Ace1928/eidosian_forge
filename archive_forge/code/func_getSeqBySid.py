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
def getSeqBySid(self, domain):
    """Get the seq record of a given domain from its sid."""
    if self.db_handle is None:
        return self.fasta_dict[domain].seq
    else:
        cur = self.db_handle.cursor()
        cur.execute('SELECT seq FROM astral WHERE sid=%s', domain)
        return Seq(cur.fetchone()[0])