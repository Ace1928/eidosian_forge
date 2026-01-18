import itertools
import functools
import importlib.util
@default_to_neutral_style
def plot_history_size_footprint(self, log=None, figsize=(8, 2), color='purple', alpha=0.5, rasterize=4096, rasterize_dpi=300, ax=None, show_and_close=True):
    """Plot the memory footprint throughout this computation.

    Parameters
    ----------
    log : None or int, optional
        If not None, display the sizes in base ``log``.
    figsize : tuple, optional
        Size of the figure.
    color : str, optional
        Color of the line.
    alpha : float, optional
        Alpha of the line.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, will be created if not provided.
    return_fig : bool, optional
        If True, return the figure object, else just show and close it.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    y = np.array(self.history_size_footprint())
    if log:
        y = np.log2(y) / np.log2(log)
        ylabel = f'$\\log_{log}[total size]$'
    else:
        ylabel = 'total size'
    x = np.arange(y.size)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(rasterize_dpi)
    else:
        fig = None
    if isinstance(rasterize, (float, int)):
        rasterize = y.size > rasterize
    if rasterize:
        ax.set_rasterization_zorder(0)
    ax.fill_between(x, 0, y, alpha=alpha, color=color, zorder=-1)
    if fig is not None:
        ax.grid(True, c=(0.95, 0.95, 0.95), which='both')
        ax.set_axisbelow(True)
        ax.set_xlim(0, np.max(x))
        ax.set_ylim(0, np.max(y))
        ax.set_ylabel(ylabel)
    if fig is not None and show_and_close:
        plt.show()
        plt.close(fig)
    return (fig, ax)