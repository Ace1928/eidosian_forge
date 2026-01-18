from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_avg_fst_pair_locus(self, locus):
    """Calculate Allele size-base average Fis for all population pairs of the given locus."""
    if len(self.__fst_pair_locus) == 0:
        iter = self._controller.calc_fst_pair(self._fname)[0]
        for locus_info in iter:
            self.__fst_pair_locus[locus_info[0]] = locus_info[1]
    return self.__fst_pair_locus[locus]