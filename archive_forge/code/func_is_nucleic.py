import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def is_nucleic(residue, standard=False):
    """Return True if residue object/string is a nucleic acid.

    :param residue: a L{Residue} object OR a three letter code
    :type residue: L{Residue} or string

    :param standard: flag to check for the 8 (DNA + RNA) canonical bases.
        Default is False.
    :type standard: boolean

    >>> is_nucleic('DA ')
    True

    >>> is_nucleic('A  ')
    True

    Known three letter codes for modified nucleotides are supported,

    >>> is_nucleic('A2L')
    True
    >>> is_nucleic('A2L', standard=True)
    False
    """
    if not isinstance(residue, str):
        residue = f'{residue.get_resname():<3s}'
    residue = residue.upper()
    if standard:
        return residue in nucleic_letters_3to1
    else:
        return residue in nucleic_letters_3to1_extended