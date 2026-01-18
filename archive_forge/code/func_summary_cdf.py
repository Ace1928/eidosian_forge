from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable
def summary_cdf(self, idx, frac, crit, varnames=None, title=None):
    """summary table for cumulative density function


        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        frac : array_like, float
            probabilities for which
        crit : array_like
            values for which cdf is calculated
        varnames : None, or list of strings
            optional list of variable names, same length as idx

        Returns
        -------
        table : instance of SimpleTable
            use `print(table` to see results


        """
    idx = np.atleast_1d(idx)
    mml = []
    for i in range(len(idx)):
        mml.append(self.cdf(crit[:, i], [idx[i]])[1].ravel())
    mmlar = np.column_stack([frac] + mml)
    if title:
        title = title + ' Probabilites'
    else:
        title = 'Probabilities'
    if varnames is None:
        varnames = ['var%d' % i for i in range(mmlar.shape[1] - 1)]
    headers = ['prob'] + varnames
    return SimpleTable(mmlar, txt_fmt={'data_fmts': ['%#6.3f'] + ['%#10.4f'] * (np.array(mml).shape[1] - 1)}, title=title, headers=headers)