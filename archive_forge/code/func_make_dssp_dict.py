import re
import os
from io import StringIO
import subprocess
import warnings
from Bio.PDB.AbstractPropertyMap import AbstractResiduePropertyMap
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1, residue_sasa_scales
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
def make_dssp_dict(filename):
    """DSSP dictionary mapping identifiers to properties.

    Return a DSSP dictionary that maps (chainid, resid) to
    aa, ss and accessibility, from a DSSP file.

    Parameters
    ----------
    filename : string
        the DSSP output file

    """
    with open(filename) as handle:
        return _make_dssp_dict(handle)