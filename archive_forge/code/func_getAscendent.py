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
def getAscendent(self, node_type):
    """Return the ancenstor node of the given type, or None.

        Node type can be a two letter code or longer description,
        e.g. 'fa' or 'family'.
        """
    if node_type in _nodetype_to_code:
        node_type = _nodetype_to_code[node_type]
    if self.scop:
        return self.scop.getAscendentFromSQL(self, node_type)
    else:
        n = self
        if n.type == node_type:
            return None
        while n.type != node_type:
            if n.type == 'ro':
                return None
            n = n.getParent()
        return n