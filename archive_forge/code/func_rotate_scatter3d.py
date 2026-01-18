from .. import select
from .. import utils
from .._lazyload import matplotlib as mpl
from . import colors
from .tools import create_colormap
from .tools import create_normalize
from .tools import generate_colorbar
from .tools import generate_legend
from .tools import label_axis
from .utils import _get_figure
from .utils import _in_ipynb
from .utils import _is_color_array
from .utils import _with_default
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
import numbers
import numpy as np
import pandas as pd
import warnings
@utils._with_pkg(pkg='matplotlib', min_version=3)
def rotate_scatter3d(data, filename=None, rotation_speed=30, fps=10, ax=None, figsize=None, elev=None, ipython_html='jshtml', dpi=None, **kwargs):
    """Create a rotating 3D scatter plot.

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, `phate.PHATE` or `scanpy.AnnData`
        Input data. Only the first three dimensions are used.
    filename : str, optional (default: None)
        If not None, saves a .gif or .mp4 with the output
    rotation_speed : float, optional (default: 30)
        Speed of axis rotation, in degrees per second
    fps : int, optional (default: 10)
        Frames per second. Increase this for a smoother animation
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    elev : int, optional (default: None)
        Elevation angle of viewpoint from horizontal, in degrees
    ipython_html : {'html5', 'jshtml'}
        which html writer to use if using a Jupyter Notebook
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **kwargs : keyword arguments
        See :~func:`scprep.plot.scatter3d`.

    Returns
    -------
    ani : `matplotlib.animation.FuncAnimation`
        animation object

    Examples
    --------
    >>> import scprep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.normal(0, 1, [200, 3])
    >>> # Continuous color vector
    >>> colors = data[:, 0]
    >>> scprep.plot.rotate_scatter3d(data, c=colors, filename="animation.gif")
    >>> # Discrete color vector with custom colormap
    >>> colors = np.random.choice(['a','b'], data.shape[0], replace=True)
    >>> data[colors == 'a'] += 5
    >>> scprep.plot.rotate_scatter3d(
            data,
            c=colors,
            cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'},
            filename="animation.mp4"
        )
    """
    if _in_ipynb():
        mpl.rc('animation', html=ipython_html)
    if filename is not None:
        if filename.endswith('.gif'):
            writer = 'imagemagick'
        elif filename.endswith('.mp4'):
            writer = 'ffmpeg'
        else:
            raise ValueError('filename must end in .gif or .mp4. Got {}'.format(filename))
    fig, ax, show_fig = _get_figure(ax, figsize, subplot_kw={'projection': '3d'})
    degrees_per_frame = rotation_speed / fps
    frames = int(round(360 / degrees_per_frame))
    degrees_per_frame = 360 / frames
    interval = 1000 * degrees_per_frame / rotation_speed
    scatter3d(data, ax=ax, elev=elev, **kwargs)
    azim = ax.azim

    def init():
        return ax

    def animate(i):
        ax.view_init(azim=azim + i * degrees_per_frame, elev=elev)
        return ax
    ani = mpl.animation.FuncAnimation(fig, animate, init_func=init, frames=range(frames), interval=interval, blit=False)
    if filename is not None:
        ani.save(filename, writer=writer, dpi=dpi)
    if _in_ipynb():
        plt.close(fig)
    elif show_fig:
        show(fig)
    return ani