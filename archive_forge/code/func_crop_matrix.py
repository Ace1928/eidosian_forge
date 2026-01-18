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
def crop_matrix(self, matrix=None, delete=(), exclude=()):
    """Return a matrix without deleted taxa and excluded characters."""
    if not matrix:
        matrix = self.matrix
    if [t for t in delete if not self._check_taxlabels(t)]:
        raise NexusError(f'Unknown taxa: {', '.join(set(delete).difference(self.taxlabels))}')
    if exclude != []:
        undelete = [t for t in self.taxlabels if t in matrix and t not in delete]
        if not undelete:
            return {}
        m = [str(matrix[k]) for k in undelete]
        sitesm = [s for i, s in enumerate(zip(*m)) if i not in exclude]
        if sitesm == []:
            return {t: Seq('') for t in undelete}
        else:
            m = [Seq(s) for s in (''.join(x) for x in zip(*sitesm))]
            return dict(zip(undelete, m))
    else:
        return {t: matrix[t] for t in self.taxlabels if t in matrix and t not in delete}