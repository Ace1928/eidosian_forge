import numpy as np
from scipy import stats
import pandas as pd
def get_prediction_linear(self, exog=None, transform=True, row_labels=None, pred_kwds=None, index=None):
    """
    Compute prediction results for linear prediction.

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
    k1 = exog.shape[1]
    if len(self.params > k1):
        index = np.arange(k1)
    else:
        index = None
    covb = self.cov_params(column=index)
    var_pred = (exog * np.dot(covb, exog.T).T).sum(1)
    pred_kwds_linear = pred_kwds.copy()
    pred_kwds_linear['which'] = 'linear'
    predicted = self.model.predict(self.params, exog, **pred_kwds_linear)
    dist = ['norm', 't'][self.use_t]
    res = PredictionResultsBase(predicted, var_pred, df=self.df_resid, dist=dist, row_labels=row_labels)
    return res