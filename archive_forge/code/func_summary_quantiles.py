from statsmodels.compat.python import lrange
import numpy as np
from statsmodels.iolib.table import SimpleTable
def summary_quantiles(self, idx, distppf, frac=[0.01, 0.025, 0.05, 0.1, 0.975], varnames=None, title=None):
    """summary table for quantiles (critical values)

        Parameters
        ----------
        idx : None or list of integers
            List of indices into the Monte Carlo results (columns) that should
            be used in the calculation
        distppf : callable
            probability density function of reference distribution
            TODO: use `crit` values instead or additional, see summary_cdf
        frac : array_like, float
            probabilities for which
        varnames : None, or list of strings
            optional list of variable names, same length as idx

        Returns
        -------
        table : instance of SimpleTable
            use `print(table` to see results

        """
    idx = np.atleast_1d(idx)
    quant, mcq = self.quantiles(idx, frac=frac)
    crit = distppf(np.atleast_2d(quant).T)
    mml = []
    for i, ix in enumerate(idx):
        mml.extend([mcq[:, i], crit[:, i]])
    mmlar = np.column_stack([quant] + mml)
    if title:
        title = title + ' Quantiles (critical values)'
    else:
        title = 'Quantiles (critical values)'
    if varnames is None:
        varnames = ['var%d' % i for i in range(mmlar.shape[1] // 2)]
    headers = ['\nprob'] + ['{}\n{}'.format(i, t) for i in varnames for t in ['mc', 'dist']]
    return SimpleTable(mmlar, txt_fmt={'data_fmts': ['%#6.3f'] + ['%#10.4f'] * (mmlar.shape[1] - 1)}, title=title, headers=headers)