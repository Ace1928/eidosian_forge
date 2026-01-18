import warnings
from Bio.Data.PDBData import nucleic_letters_3to1
from Bio.Data.PDBData import nucleic_letters_3to1_extended
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.Seq import Seq
def get_sequence(self):
    """Return the AA sequence as a Seq object.

        :return: polypeptide sequence
        :rtype: L{Seq}
        """
    s = ''.join((protein_letters_3to1_extended.get(res.get_resname(), 'X') for res in self))
    return Seq(s)