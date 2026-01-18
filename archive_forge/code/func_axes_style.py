import functools
import matplotlib as _mpl
def axes_style(style=None, rc=None):
    """
    Return a parameter dict for the aesthetic style of the plots

    This affects things like the color of the axes, whether a grid is
    enabled by default, and other aesthetic elements.

    This function returns an object that can be used in a `with` statement
    to temporarily change the style parameters.

    Parameters
    ----------
    style : "darkgrid" | "whitegrid" | "dark" | "white" | "ticks" | dict | None
        A dictionary of parameters or the name of a preconfigured set.
    rc : dict
        Parameter mappings to override the values in the preset seaborn
        style dictionaries. This only updates parameters that are
        considered part of the style definition.

    Examples
    --------
    >>> st = axes_style("whitegrid")

    >>> set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    >>> import matplotlib.pyplot as plt
    >>> with axes_style("white"):
    ...     f, ax = plt.subplots()
    ...     ax.plot(x, y)               # doctest: +SKIP

    See Also
    --------
    set_style : set the matplotlib parameters for a seaborn theme
    plotting_context : return a parameter dict to to scale plot elements
    color_palette : define the color palette for a plot

    """
    if style is None:
        style_dict = {k: mpl.rcParams[k] for k in _style_keys}
    elif isinstance(style, dict):
        style_dict = style
    else:
        styles = ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']
        if style not in styles:
            raise ValueError(f'style must be one of {', '.join(styles)}')
        dark_gray = '.15'
        light_gray = '.8'
        style_dict = {'figure.facecolor': 'white', 'axes.labelcolor': dark_gray, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.color': dark_gray, 'ytick.color': dark_gray, 'axes.axisbelow': True, 'grid.linestyle': '-', 'text.color': dark_gray, 'font.family': ['sans-serif'], 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'], 'lines.solid_capstyle': 'round', 'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'image.cmap': 'Greys', 'xtick.top': False, 'ytick.right': False}
        if 'grid' in style:
            style_dict.update({'axes.grid': True})
        else:
            style_dict.update({'axes.grid': False})
        if style.startswith('dark'):
            style_dict.update({'axes.facecolor': '#EAEAF2', 'axes.edgecolor': 'white', 'grid.color': 'white', 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True})
        elif style == 'whitegrid':
            style_dict.update({'axes.facecolor': 'white', 'axes.edgecolor': light_gray, 'grid.color': light_gray, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True})
        elif style in ['white', 'ticks']:
            style_dict.update({'axes.facecolor': 'white', 'axes.edgecolor': dark_gray, 'grid.color': light_gray, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True})
        if style == 'ticks':
            style_dict.update({'xtick.bottom': True, 'ytick.left': True})
        else:
            style_dict.update({'xtick.bottom': False, 'ytick.left': False})
    style_dict = {k: v for k, v in style_dict.items() if k in _style_keys}
    if rc is not None:
        rc = {k: v for k, v in rc.items() if k in _style_keys}
        style_dict.update(rc)
    style_object = _AxesStyle(style_dict)
    return style_object