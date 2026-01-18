from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_avg_fis(self):
    """Calculate identity-base average Fis."""
    return self._controller.calc_diversities_fis_with_identity(self._fname)[1]