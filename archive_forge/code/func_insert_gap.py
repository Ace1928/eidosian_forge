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
def insert_gap(self, pos, n=1, leftgreedy=False):
    """Add a gap into the matrix and adjust charsets and partitions.

        pos=0: first position
        pos=nchar: last position
        """

    def _adjust(set, x, d, leftgreedy=False):
        """Adjust character sets if gaps are inserted (PRIVATE).

            Takes care of new gaps within a coherent character set.
            """
        set.sort()
        addpos = 0
        for i, c in enumerate(set):
            if c >= x:
                set[i] = c + d
            if c == x:
                if leftgreedy or (i > 0 and set[i - 1] == c - 1):
                    addpos = i
        if addpos > 0:
            set[addpos:addpos] = list(range(x, x + d))
        return set
    if pos < 0 or pos > self.nchar:
        raise NexusError('Illegal gap position: %d' % pos)
    if n == 0:
        return
    sitesm = list(zip(*(str(self.matrix[t]) for t in self.taxlabels)))
    sitesm[pos:pos] = [['-'] * len(self.taxlabels)] * n
    mapped = [''.join(x) for x in zip(*sitesm)]
    listed = [(taxon, Seq(mapped[i])) for i, taxon in enumerate(self.taxlabels)]
    self.matrix = dict(listed)
    self.nchar += n
    for i, s in self.charsets.items():
        self.charsets[i] = _adjust(s, pos, n, leftgreedy=leftgreedy)
    for p in self.charpartitions:
        for sp, s in self.charpartitions[p].items():
            self.charpartitions[p][sp] = _adjust(s, pos, n, leftgreedy=leftgreedy)
    self.charlabels = self._adjust_charlabels(insert=[pos] * n)
    return self.charlabels