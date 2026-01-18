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
def terminal_gap_to_missing(self, missing=None, skip_n=True):
    """Replace all terminal gaps with missing character.

        Mixtures like ???------??------- are properly resolved.
        """
    if not missing:
        missing = self.missing
    replace = [self.missing, self.gap]
    if not skip_n:
        replace.extend(['n', 'N'])
    for taxon in self.taxlabels:
        sequence = str(self.matrix[taxon])
        length = len(sequence)
        start, end = get_start_end(sequence, skiplist=replace)
        if start == -1 and end == -1:
            sequence = missing * length
        else:
            sequence = sequence[:end + 1] + missing * (length - end - 1)
            sequence = start * missing + sequence[start:]
        if length != len(sequence):
            raise RuntimeError('Illegal sequence manipulation in Nexus.terminal_gap_to_missing in taxon %s' % taxon)
        self.matrix[taxon] = Seq(sequence)