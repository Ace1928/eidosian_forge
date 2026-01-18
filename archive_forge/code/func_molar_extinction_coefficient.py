import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def molar_extinction_coefficient(self):
    """Calculate the molar extinction coefficient.

        Calculates the molar extinction coefficient assuming cysteines
        (reduced) and cystines residues (Cys-Cys-bond)
        """
    num_aa = self.count_amino_acids()
    mec_reduced = num_aa['W'] * 5500 + num_aa['Y'] * 1490
    mec_cystines = mec_reduced + num_aa['C'] // 2 * 125
    return (mec_reduced, mec_cystines)