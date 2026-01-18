from .._lazyload import matplotlib as mpl
from . import tools
import numpy as np
def tab(n=10):
    """Create a discrete colormap with an arbitrary number of colors.

    This colormap chooses the best of the following, in order:
    - `plt.cm.tab10`
    - `plt.cm.tab20`
    - `scprep.plot.colors.tab30`
    - `scprep.plot.colors.tab40`
    - `scprep.plot.colors.tab10_continuous`

    If the number of colors required is less than the number of colors
    available, colors are selected specifically in order to reduce similarity
    between selected colors.

    Parameters
    ----------
    n : int, optional (default: 10)
        Number of required colors.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    """
    if n < 1:
        raise ValueError('Expected n >= 1. Got {}'.format(n))
    n_shades = int(np.ceil(n / 10))
    if n_shades == 1:
        cmap = mpl.cm.tab10
    elif n_shades == 2:
        cmap = mpl.cm.tab20
    elif n_shades == 3:
        cmap = tab30()
    elif n_shades == 4:
        cmap = tab40()
    else:
        cmap = tab10_continuous(n_colors=10, n_step=n_shades)
    if n > 1 and n < cmap.N:
        select_idx = np.tile(np.arange(10), n_shades) * n_shades + np.repeat(np.arange(n_shades), 10)
        cmap = mpl.colors.ListedColormap(np.array(cmap.colors)[select_idx[:n]])
    return cmap