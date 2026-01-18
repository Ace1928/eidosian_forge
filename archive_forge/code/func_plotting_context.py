import functools
import matplotlib as _mpl
def plotting_context(context=None, font_scale=1, rc=None):
    """
    Return a parameter dict to scale elements of the figure

    This affects things like the size of the labels, lines, and other
    elements of the plot, but not the overall style. The base context
    is "notebook", and the other contexts are "paper", "talk", and "poster",
    which are version of the notebook parameters scaled by .8, 1.3, and 1.6,
    respectively.

    This function returns an object that can be used in a `with` statement
    to temporarily change the context parameters.

    Parameters
    ----------
    context : dict, None, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        context dictionaries. This only updates parameters that are
        considered part of the context definition.

    Examples
    --------
    >>> c = plotting_context("poster")

    >>> c = plotting_context("notebook", font_scale=1.5)

    >>> c = plotting_context("talk", rc={"lines.linewidth": 2})

    >>> import matplotlib.pyplot as plt
    >>> with plotting_context("paper"):
    ...     f, ax = plt.subplots()
    ...     ax.plot(x, y)                 # doctest: +SKIP

    See Also
    --------
    set_context : set the matplotlib parameters to scale plot elements
    axes_style : return a dict of parameters defining a figure style
    color_palette : define the color palette for a plot
    """
    if context is None:
        context_dict = {k: mpl.rcParams[k] for k in _context_keys}
    elif isinstance(context, dict):
        context_dict = context
    else:
        contexts = ['paper', 'notebook', 'talk', 'poster']
        if context not in contexts:
            raise ValueError(f'context must be in {', '.join(contexts)}')
        texts_base_context = {'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11, 'legend.title_fontsize': 12}
        base_context = {'axes.linewidth': 1.25, 'grid.linewidth': 1, 'lines.linewidth': 1.5, 'lines.markersize': 6, 'patch.linewidth': 1, 'xtick.major.width': 1.25, 'ytick.major.width': 1.25, 'xtick.minor.width': 1, 'ytick.minor.width': 1, 'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.minor.size': 4, 'ytick.minor.size': 4}
        base_context.update(texts_base_context)
        scaling = {'paper': 0.8, 'notebook': 1, 'talk': 1.5, 'poster': 2}[context]
        context_dict = {k: v * scaling for k, v in base_context.items()}
        font_keys = texts_base_context.keys()
        font_dict = {k: context_dict[k] * font_scale for k in font_keys}
        context_dict.update(font_dict)
    if rc is not None:
        rc = {k: v for k, v in rc.items() if k in _context_keys}
        context_dict.update(rc)
    context_object = _PlottingContext(context_dict)
    return context_object