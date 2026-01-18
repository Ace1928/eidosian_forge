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
def domainsClusteredByEv(self, id):
    """Get domains clustered by evalue."""
    if id not in self.EvDatasets:
        if self.db_handle:
            self.EvDatasets[id] = self.getAstralDomainsFromSQL(astralEv_to_sql[id])
        else:
            if not self.path:
                raise RuntimeError('No scopseq directory specified')
            file_prefix = 'astral-scopdom-seqres-sel-gs'
            filename = '%s-e100m-%s-%s.id' % (file_prefix, astralEv_to_file[id], self.version)
            filename = os.path.join(self.path, filename)
            self.EvDatasets[id] = self.getAstralDomainsFromFile(filename)
    return self.EvDatasets[id]