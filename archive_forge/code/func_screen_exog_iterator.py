from collections import defaultdict
import numpy as np
from statsmodels.base._penalties import SCADSmoothed
def screen_exog_iterator(self, exog_iterator):
    """
        batched version of screen exog

        This screens variables in a two step process:

        In the first step screen_exog is used on each element of the
        exog_iterator, and the batch winners are collected.

        In the second step all batch winners are combined into a new array
        of exog candidates and `screen_exog` is used to select a final
        model.

        Parameters
        ----------
        exog_iterator : iterator over ndarrays

        Returns
        -------
        res_screen_final : instance of ScreeningResults
            This is the instance returned by the second round call to
            `screen_exog`. Additional attributes are added to provide
            more information about the batched selection process.
            The index of final nonzero variables is
            `idx_nonzero_batches` which is a 2-dimensional array with batch
            index in the first column and variable index within batch in the
            second column. They can be used jointly as index for the data
            in the exog_iterator.
            see ScreeningResults for a full description
        """
    k_keep = self.k_keep
    res_idx = []
    exog_winner = []
    exog_idx = []
    for ex in exog_iterator:
        res_screen = self.screen_exog(ex, maxiter=20)
        res_idx.append(res_screen.idx_nonzero)
        exog_winner.append(ex[:, res_screen.idx_nonzero[k_keep:] - k_keep])
        exog_idx.append(res_screen.idx_nonzero[k_keep:] - k_keep)
    exog_winner = np.column_stack(exog_winner)
    res_screen_final = self.screen_exog(exog_winner, maxiter=20)
    exog_winner_names = ['var%d_%d' % (bidx, idx) for bidx, batch in enumerate(exog_idx) for idx in batch]
    idx_full = [(bidx, idx) for bidx, batch in enumerate(exog_idx) for idx in batch]
    ex_final_idx = res_screen_final.idx_nonzero[k_keep:] - k_keep
    final_names = np.array(exog_winner_names)[ex_final_idx]
    res_screen_final.idx_nonzero_batches = np.array(idx_full)[ex_final_idx]
    res_screen_final.exog_final_names = final_names
    history = {'idx_nonzero': res_idx, 'idx_exog': exog_idx}
    res_screen_final.history_batches = history
    return res_screen_final