import numpy as np
from scipy import stats
import pandas as pd
def get_prediction_monotonic(self, exog=None, transform=True, row_labels=None, link=None, pred_kwds=None, index=None):
    """
    Compute prediction results when endpoint transformation is valid.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.
    link : instance of link function
        If no link function is provided, then the ``mmodel.family.link` is
        used.
    pred_kwargs :
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models.
        See the predict method of the model for the details.
    index : slice or array-index
        Is used to select rows and columns of cov_params, if the prediction
        function only depends on a subset of parameters.

    Returns
    -------
    prediction_results : PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction.
    """
    exog, row_labels = _get_exog_predict(self, exog=exog, transform=transform, row_labels=row_labels)
    if pred_kwds is None:
        pred_kwds = {}
    if link is None:
        link = self.model.family.link
    func_deriv = link.inverse_deriv
    covb = self.cov_params(column=index)
    linpred_var = (exog * np.dot(covb, exog.T).T).sum(1)
    pred_kwds_linear = pred_kwds.copy()
    pred_kwds_linear['which'] = 'linear'
    linpred = self.model.predict(self.params, exog, **pred_kwds_linear)
    predicted = self.model.predict(self.params, exog, **pred_kwds)
    link_deriv = func_deriv(linpred)
    var_pred = link_deriv ** 2 * linpred_var
    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsMonotonic(predicted, var_pred, df=self.df_resid, dist=dist, row_labels=row_labels, linpred=linpred, linpred_se=np.sqrt(linpred_var), func=link.inverse, deriv=func_deriv)
    return res