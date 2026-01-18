from ..units import latex_of_unit, is_unitless, to_unitless, unit_of
from ..printing import number_to_scientific_latex
def irls(x, y, w_cb=lambda x, y, b, c: x ** 0, itermax=16, rmsdwtol=1e-08):
    """Iteratively reweighted least squares

    Parameters
    ----------
    x : array_like
    y : array_like
    w_cb : callbable
        Weight callback, signature ``(x, y, beta, cov) -> weight``.
        Predefined:
            - ``irls.ones``: unit weights (default)
            - ``irls.exp``: :math:`\\mathrm{e}^{-\\beta_2 x}`
            - ``irls.gaussian``: :math:`\\mathrm{e}^{-\\beta_2 x^2}`
            - ``irls.abs_residuals``: :math:`\\lvert \\beta_1 + \\beta_2 x - y \\rvert`
    itermax : int
    rmsdwtol : float
    plot_cb : callble
        See :func:`least_squares`
    plot_cb_kwargs : dict
        See :func:`least_squares`

    Returns
    -------
    beta : length-2 array
        parameters
    cov : 2x2 array
        variance-covariance matrix
    info : dict
        Contains
           - success : bool
           - niter : int
           - weights : list of weighting arrays

    # Examples
    # --------
    # beta, cov, info = irls([1, 2, 3], [3, 2.5, 2.1], irls.abs_residuals)

    """
    if itermax < 1:
        raise ValueError('itermax must be >= 1')
    weights = []
    x, y = (np.asarray(x), np.asarray(y))
    w = np.ones_like(x)
    rmsdw = np.inf
    ii = 0
    while rmsdw > rmsdwtol and ii < itermax:
        weights.append(w)
        beta, cov, r2 = least_squares(x, y, w)
        old_w = w.copy()
        w = w_cb(x, y, beta, cov)
        rmsdw = np.sqrt(np.mean(np.square(w - old_w)))
        ii += 1
    return (beta, cov, {'weights': weights, 'niter': ii, 'success': ii < itermax})