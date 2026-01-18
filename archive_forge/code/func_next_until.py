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
def next_until(self, target):
    """Iterate over the NEXUS file until a target character is reached."""
    for t in target:
        try:
            pos = self.buffer.index(t)
        except ValueError:
            pass
        else:
            found = ''.join(self.buffer[:pos])
            self.buffer = self.buffer[pos:]
            return found
    else:
        return None