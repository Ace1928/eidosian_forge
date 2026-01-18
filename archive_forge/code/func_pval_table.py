from statsmodels.compat.python import lzip
import numpy as np
from statsmodels.tools.testing import Holder
def pval_table(self):
    """create a (n_levels, n_levels) array with corrected p_values

        this needs to improve, similar to R pairwise output
        """
    k = self.n_levels
    pvals_mat = np.zeros((k, k))
    pvals_mat[lzip(*self.all_pairs)] = self.pval_corrected()
    return pvals_mat