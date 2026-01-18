from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_f_stats(self, locus_name):
    """Return F stats for a locus.

        Returns Fis(CW), Fst, Fit, Qintra, Qinter
        """
    loci_iter = self._controller.calc_fst_all(self._fname)[1]
    for name, fis, fst, fit, qintra, qinter in loci_iter:
        if name == locus_name:
            return (fis, fst, fit, qintra, qinter)