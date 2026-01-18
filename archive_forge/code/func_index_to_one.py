import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def index_to_one(index):
    """Index to corresponding one letter amino acid name.

    >>> index_to_one(0)
    'A'
    >>> index_to_one(19)
    'Y'
    """
    return dindex_to_1[index]