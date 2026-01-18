from functools import reduce
import copy
import math
import random
import sys
import warnings
from Bio import File
from Bio.Data import IUPACData
from Bio.Seq import Seq
from Bio import BiopythonDeprecationWarning, BiopythonWarning
from Bio.Nexus.StandardData import StandardData
from Bio.Nexus.Trees import Tree
def export_phylip(self, filename=None):
    """Write matrix into a PHYLIP file.

        Note that this writes a relaxed PHYLIP format file, where the names
        are not truncated, nor checked for invalid characters.
        """
    if not filename:
        if '.' in self.filename and self.filename.split('.')[-1].lower() in ['paup', 'nexus', 'nex', 'dat']:
            filename = '.'.join(self.filename.split('.')[:-1]) + '.phy'
        else:
            filename = self.filename + '.phy'
    with open(filename, 'w') as fh:
        fh.write('%d %d\n' % (self.ntax, self.nchar))
        for taxon in self.taxlabels:
            fh.write(f'{safename(taxon)} {self.matrix[taxon]!s}\n')
    return filename