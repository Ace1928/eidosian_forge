from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_avg_fst_pair(self):
    """Calculate Allele size-base average Fis for all population pairs."""
    return self._controller.calc_fst_pair(self._fname)[1]