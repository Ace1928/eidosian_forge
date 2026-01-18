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
def residue_depth(residue, surface):
    """Residue depth as average depth of all its atoms.

    Return average distance to surface for all atoms in a residue,
    ie. the residue depth.
    """
    atom_list = residue.get_unpacked_list()
    length = len(atom_list)
    d = 0
    for atom in atom_list:
        coord = atom.get_coord()
        d = d + min_dist(coord, surface)
    return d / length