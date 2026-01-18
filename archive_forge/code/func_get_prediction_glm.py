import numpy as np
from scipy import stats
import pandas as pd
def get_prediction_glm(self, exog=None, transform=True, row_labels=None, linpred=None, link=None, pred_kwds=None):
    """
    Compute prediction results for GLM compatible models.

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
    linpred : linear prediction instance
        Instance of linear prediction results used for confidence intervals
        based on endpoint transformation.
    link : instance of link function
        If no link function is provided, then the `model.family.link` is used.
    pred_kwds : dict
        Some models can take additional keyword arguments, such as offset or
        additional exog in multi-part models. See the predict method of the
        model for the details.

    Returns
    -------
    prediction_results : generalized_linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """
    exog, row_labels = _get_exog_predict(self, exog=exog, transform=transform, row_labels=row_labels)
    if pred_kwds is None:
        pred_kwds = {}
    predicted_mean = self.model.predict(self.params, exog, **pred_kwds)
    covb = self.cov_params()
    link_deriv = self.model.family.link.inverse_deriv(linpred.predicted_mean)
    var_pred_mean = link_deriv ** 2 * (exog * np.dot(covb, exog.T).T).sum(1)
    var_resid = self.scale
    if self.cov_type == 'fixed scale':
        var_resid = self.cov_kwds['scale']
    dist = ['norm', 't'][self.use_t]
    return PredictionResultsMean(predicted_mean, var_pred_mean, var_resid, df=self.df_resid, dist=dist, row_labels=row_labels, linpred=linpred, link=link)