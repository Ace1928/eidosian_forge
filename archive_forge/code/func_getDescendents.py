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
def getDescendents(self, node_type):
    """Return a list of all descendant nodes of the given type.

        Node type can be a two letter code or longer description,
        e.g. 'fa' or 'family'.
        """
    if node_type in _nodetype_to_code:
        node_type = _nodetype_to_code[node_type]
    nodes = [self]
    if self.scop:
        return self.scop.getDescendentsFromSQL(self, node_type)
    while nodes[0].type != node_type:
        if nodes[0].type == 'px':
            return []
        child_list = []
        for n in nodes:
            for child in n.getChildren():
                child_list.append(child)
            nodes = child_list
    return nodes