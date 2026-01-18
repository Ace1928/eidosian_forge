from ..units import latex_of_unit, is_unitless, to_unitless, unit_of
from ..printing import number_to_scientific_latex
def avg_params(opt_params, cov_params):
    """Calculates the average parameters from a set of regression parameters.

    Parameters
    ----------
    opt_params : array_like
        of shape (nfits, nparams)
    cov_params : array_like
        of shape (nfits, nparams, nparams)

    Returns
    -------
    avg_beta: weighted average of parameters
    var_avg_beta: variance-covariance matrix

    """
    opt_params = np.asarray(opt_params)
    cov_params = np.asarray(cov_params)
    var_beta = np.vstack((cov_params[:, 0, 0], cov_params[:, 1, 1])).T
    avg_beta, sum_of_weights = np.average(opt_params, axis=0, weights=1 / var_beta, returned=True)
    var_avg_beta = np.sum(np.square(opt_params - avg_beta) / var_beta, axis=0) / ((avg_beta.shape[0] - 1) * sum_of_weights)
    return (avg_beta, var_avg_beta)