import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def instability_index(self):
    """Calculate the instability index according to Guruprasad et al 1990.

        Implementation of the method of Guruprasad et al. 1990 to test a
        protein for stability. Any value above 40 means the protein is unstable
        (has a short half life).

        See: Guruprasad K., Reddy B.V.B., Pandit M.W.
        Protein Engineering 4:155-161(1990).
        """
    index = ProtParamData.DIWV
    score = 0.0
    for i in range(self.length - 1):
        this, next = self.sequence[i:i + 2]
        dipeptide_value = index[this][next]
        score += dipeptide_value
    return 10.0 / self.length * score