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
def smprint(self, name='your_name_here'):
    """Print a stepmatrix."""
    matrix = 'usertype %s stepmatrix=%d\n' % (name, len(self.symbols))
    matrix += f'        {'        '.join(self.symbols)}\n'
    for x in self.symbols:
        matrix += '[%s]'.ljust(8) % x
        for y in self.symbols:
            if x == y:
                matrix += ' .       '
            else:
                if x > y:
                    x1, y1 = (y, x)
                else:
                    x1, y1 = (x, y)
                if self.data[x1 + y1] == 0:
                    matrix += 'inf.     '
                else:
                    matrix += '%2.2f'.ljust(10) % self.data[x1 + y1]
        matrix += '\n'
    matrix += ';\n'
    return matrix