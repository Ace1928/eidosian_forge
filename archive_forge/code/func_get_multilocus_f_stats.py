from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_multilocus_f_stats(self):
    """Return the multilocus F stats.

        Explain averaging.
        Returns Fis(CW), Fst, Fit
        """
    return self._controller.calc_fst_all(self._fname)[0]