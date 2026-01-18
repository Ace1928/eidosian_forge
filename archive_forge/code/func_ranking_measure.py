from collections import defaultdict
import numpy as np
from statsmodels.base._penalties import SCADSmoothed
def ranking_measure(self, res_pen, exog, keep=None):
    """compute measure for ranking exog candidates for inclusion
        """
    endog = self.endog
    if self.ranking_project:
        assert res_pen.model.exog.shape[1] == len(keep)
        ex_incl = res_pen.model.exog[:, keep]
        exog = exog - ex_incl.dot(np.linalg.pinv(ex_incl).dot(exog))
    if self.ranking_attr == 'predicted_poisson':
        p = res_pen.params.copy()
        if keep is not None:
            p[~keep] = 0
        predicted = res_pen.model.predict(p)
        resid_factor = (endog - predicted) / np.sqrt(predicted)
    elif self.ranking_attr[:6] == 'model.':
        attr = self.ranking_attr.split('.')[1]
        resid_factor = getattr(res_pen.model, attr)(res_pen.params)
        if resid_factor.ndim == 2:
            resid_factor = resid_factor[:, 0]
        mom_cond = np.abs(resid_factor.dot(exog)) ** 2
    else:
        resid_factor = getattr(res_pen, self.ranking_attr)
        mom_cond = np.abs(resid_factor.dot(exog)) ** 2
    return mom_cond