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
def gaponly(self, include_missing=False):
    """Return gap-only sites."""
    gap = set(self.gap)
    if include_missing:
        gap.add(self.missing)
    sitesm = zip(*(str(self.matrix[t]) for t in self.taxlabels))
    return [i for i, site in enumerate(sitesm) if set(site).issubset(gap)]