from numpy.testing import assert_equal
import numpy as np
def contrast_allpairs(nm):
    """contrast or restriction matrix for all pairs of nm variables

    Parameters
    ----------
    nm : int

    Returns
    -------
    contr : ndarray, 2d, (nm*(nm-1)/2, nm)
       contrast matrix for all pairwise comparisons

    """
    contr = []
    for i in range(nm):
        for j in range(i + 1, nm):
            contr_row = np.zeros(nm)
            contr_row[i] = 1
            contr_row[j] = -1
            contr.append(contr_row)
    return np.array(contr)