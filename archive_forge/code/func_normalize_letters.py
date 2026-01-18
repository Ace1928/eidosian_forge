from copy import copy
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.SCOP.Residues import Residues
def normalize_letters(one_letter_code):
    """Convert RAF one-letter amino acid codes into IUPAC standard codes.

    Letters are uppercased, and "." ("Unknown") is converted to "X".
    """
    if one_letter_code == '.':
        return 'X'
    else:
        return one_letter_code.upper()