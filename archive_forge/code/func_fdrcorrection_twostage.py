import numpy as np
from statsmodels.stats._knockoff import RegressionFDR
def fdrcorrection_twostage(pvals, alpha=0.05, method='bky', maxiter=1, iter=None, is_sorted=False):
    """(iterated) two stage linear step-up procedure with estimation of number of true
    hypotheses

    Benjamini, Krieger and Yekuteli, procedure in Definition 6

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : {'bky', 'bh')
        see Notes for details

        * 'bky' - implements the procedure in Definition 6 of Benjamini, Krieger
           and Yekuteli 2006
        * 'bh' - the two stage method of Benjamini and Hochberg

    maxiter : int or bool
        Maximum number of iterations.
        maxiter=1 (default) corresponds to the two stage method.
        maxiter=-1 corresponds to full iterations which is maxiter=len(pvals).
        maxiter=0 uses only a single stage fdr correction using a 'bh' or 'bky'
        prior fraction of assumed true hypotheses.
        Boolean maxiter is allowed for backwards compatibility with the
        deprecated ``iter`` keyword.
        maxiter=False is two-stage fdr (maxiter=1)
        maxiter=True is full iteration (maxiter=-1 or maxiter=len(pvals))

        .. versionadded:: 0.14

            Replacement for ``iter`` with additional features.

    iter : bool
        ``iter`` is deprecated use ``maxiter`` instead.
        If iter is True, then only one iteration step is used, this is the
        two-step method.
        If iter is False, then iterations are stopped at convergence which
        occurs in a finite number of steps (at most len(pvals) steps).

        .. deprecated:: 0.14

            Use ``maxiter`` instead of ``iter``.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : ndarray
        pvalues adjusted for multiple hypotheses testing to limit FDR
    m0 : int
        ntest - rej, estimated number of true (not rejected) hypotheses
    alpha_stages : list of floats
        A list of alphas that have been used at each stage

    Notes
    -----
    The returned corrected p-values are specific to the given alpha, they
    cannot be used for a different alpha.

    The returned corrected p-values are from the last stage of the fdr_bh
    linear step-up procedure (fdrcorrection0 with method='indep') corrected
    for the estimated fraction of true hypotheses.
    This means that the rejection decision can be obtained with
    ``pval_corrected <= alpha``, where ``alpha`` is the original significance
    level.
    (Note: This has changed from earlier versions (<0.5.0) of statsmodels.)

    BKY described several other multi-stage methods, which would be easy to implement.
    However, in their simulation the simple two-stage method (with iter=False) was the
    most robust to the presence of positive correlation

    TODO: What should be returned?

    """
    pvals = np.asarray(pvals)
    if iter is not None:
        import warnings
        msg = 'iter keyword is deprecated, use maxiter keyword instead.'
        warnings.warn(msg, FutureWarning)
    if iter is False:
        maxiter = 1
    elif iter is True or maxiter in [-1, None]:
        maxiter = len(pvals)
    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals = np.take(pvals, pvals_sortind)
    ntests = len(pvals)
    if method == 'bky':
        fact = 1.0 + alpha
        alpha_prime = alpha / fact
    elif method == 'bh':
        fact = 1.0
        alpha_prime = alpha
    else:
        raise ValueError("only 'bky' and 'bh' are available as method")
    alpha_stages = [alpha_prime]
    rej, pvalscorr = fdrcorrection(pvals, alpha=alpha_prime, method='indep', is_sorted=True)
    r1 = rej.sum()
    if r1 == 0 or r1 == ntests:
        reject = rej
        pvalscorr *= fact
        ri = r1
    else:
        ri_old = ri = r1
        ntests0 = ntests
        for it in range(maxiter):
            ntests0 = 1.0 * ntests - ri_old
            alpha_star = alpha_prime * ntests / ntests0
            alpha_stages.append(alpha_star)
            rej, pvalscorr = fdrcorrection(pvals, alpha=alpha_star, method='indep', is_sorted=True)
            ri = rej.sum()
            if it >= maxiter - 1 or ri == ri_old:
                break
            elif ri < ri_old:
                raise RuntimeError(' oops - should not be here')
            ri_old = ri
        pvalscorr *= ntests0 * 1.0 / ntests
        if method == 'bky':
            pvalscorr *= 1.0 + alpha
    pvalscorr[pvalscorr > 1] = 1
    if not is_sorted:
        pvalscorr_ = np.empty_like(pvalscorr)
        pvalscorr_[pvals_sortind] = pvalscorr
        del pvalscorr
        reject = np.empty_like(rej)
        reject[pvals_sortind] = rej
        return (reject, pvalscorr_, ntests - ri, alpha_stages)
    else:
        return (rej, pvalscorr, ntests - ri, alpha_stages)