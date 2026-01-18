import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def index_to_three(i):
    """Index to corresponding three letter amino acid name.

    >>> index_to_three(0)
    'ALA'
    >>> index_to_three(19)
    'TYR'
    """
    return dindex_to_3[i]