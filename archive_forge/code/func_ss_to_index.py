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
def ss_to_index(ss):
    """Secondary structure symbol to index.

    H=0
    E=1
    C=2
    """
    if ss == 'H':
        return 0
    if ss == 'E':
        return 1
    if ss == 'C':
        return 2
    assert 0