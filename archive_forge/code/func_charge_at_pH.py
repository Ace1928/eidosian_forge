import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def charge_at_pH(self, pH):
    """Calculate the charge of a protein at given pH."""
    aa_content = self.count_amino_acids()
    charge = IsoelectricPoint.IsoelectricPoint(self.sequence, aa_content)
    return charge.charge_at_pH(pH)