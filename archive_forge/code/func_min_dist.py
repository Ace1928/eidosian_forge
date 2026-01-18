import os
import subprocess
import tempfile
import warnings
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import is_aa
from Bio import BiopythonWarning
def min_dist(coord, surface):
    """Return minimum distance between coord and surface."""
    d = surface - coord
    d2 = np.sum(d * d, 1)
    return np.sqrt(min(d2))