from ..units import latex_of_unit, is_unitless, to_unitless, unit_of
from ..printing import number_to_scientific_latex
def plot_least_squares_fit(x, y, beta_vcv_r2, yerr=None, plot_cb=None, plot_cb_kwargs=None, x_unit=1, y_unit=1):
    """Performs Least-squares fit and plots data and fitted line

    Parameters
    ----------
    x : array_like
    y : array_like
    beta_vcv_r2 : tuple
        Result from :func:`least_squares_fit`.
    plot_cb : callable
        When ``None``: uses :func:`plot_fit`, when callable:
        signature ``(x, y, beta, yerr=None, fit_label_cb=lambda beta, vcv, r2: 'None') -> str``.
    plot_cb_kwargs: dict, optional
        Keyword arguments passed on to ``plot_cb`` (see :func:`plot_fit` for list of
        expected kwargs). If ``plot_cb`` is ``True`` it will be populated with defaults
        (kw_data, fit_label_cb, x_unit, y_unit).

    """
    plot_cb_kwargs = plot_cb_kwargs or {}
    if plot_cb is None:
        kw_data = plot_cb_kwargs.get('kw_data', {})
        if 'marker' not in kw_data and len(x) < 40:
            kw_data['marker'] = 'd'
        if 'ls' not in kw_data and 'linestyle' not in kw_data and (len(x) < 40):
            kw_data['ls'] = 'None'
        plot_cb_kwargs['kw_data'] = kw_data
        if 'fit_label_cb' not in plot_cb_kwargs:
            plot_cb_kwargs['fit_label_cb'] = lambda b, v, r2: '$y(x) = %s + %s \\cdot x$' % tuple(map(number_to_scientific_latex, b))
        plot_cb = plot_fit
    if 'x_unit' not in plot_cb_kwargs:
        plot_cb_kwargs['x_unit'] = x_unit
    if 'y_unit' not in plot_cb_kwargs:
        plot_cb_kwargs['y_unit'] = y_unit
    plot_cb(x, y, beta_vcv_r2[0], yerr, **plot_cb_kwargs)