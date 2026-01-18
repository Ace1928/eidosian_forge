from statsmodels.compat.python import lrange
def create_mpl_fig(fig=None, figsize=None):
    """Helper function for when multiple plot axes are needed.

    Those axes should be created in the functions they are used in, with
    ``fig.add_subplot()``.

    Parameters
    ----------
    fig : Figure, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    Figure
        If `fig` is None, the created figure.  Otherwise the input `fig` is
        returned.

    See Also
    --------
    create_mpl_ax
    """
    if fig is None:
        plt = _import_mpl()
        fig = plt.figure(figsize=figsize)
    return fig