import warnings
from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np
from . import reductions
from . import transfer_functions as tf
from .colors import Sets1to3
from .core import bypixel, Canvas
def dsshow(df, glyph, aggregator=reductions.count(), agg_hook=None, shade_hook=None, plot_width=None, plot_height=None, x_range=None, y_range=None, width_scale=1.0, height_scale=1.0, *, norm=None, cmap=None, alpha=None, vmin=None, vmax=None, color_key=Sets1to3, alpha_range=(40, 255), color_baseline=None, ax=None, fignum=None, aspect=None, **kwargs):
    """
    Display the output of a data shading pipeline applied to a dataframe.

    The plot will respond to changes in the data space bounds displayed, such
    as pan/zoom events. Both scalar and categorical pipelines are supported.

    Parameters
    ----------
    df : pandas.DataFrame, dask.DataFrame
        Dataframe to apply the datashading pipeline to.
    glyph : Glyph
        The glyph to bin by.
    aggregator : Reduction, optional, default: :class:`~.count`
        The reduction to compute per-pixel.
    agg_hook : callable, optional
        A callable that takes the computed aggregate as an argument, and
        returns another aggregate. This can be used to do preprocessing before
        the aggregate is converted to an image.
    shade_hook : callable, optional
        A callable that takes the image output of the shading pipeline, and
        returns another :class:`~.Image` object. See :func:`~.dynspread` and
        :func:`~.spread` for examples.
    plot_width, plot_height : int, optional
        Grid dimensions, i.e. the width and height of the output aggregates in
        pixels. Default is to use the native width and height dimensions of
        the axes bounding box.
    x_range, y_range : pair of float, optional
        A tuple representing the initial bounds inclusive space ``[min, max]``
        along the axis. If None, the initial bounds will encompass all of the
        data along the axis.
    height_scale : float, optional
        Factor by which to scale the height of the image in pixels relative to
        the height of the display space in pixels.
    width_scale : float, optional
        Factor by which to scale the width of the image in pixels relative to
        the width of the display space in pixels.
    norm : str or :class:`matplotlib.colors.Normalize`, optional
        For scalar aggregates, a matplotlib norm to normalize the
        aggregate data to [0, 1] before colormapping. The datashader arguments
        'linear', 'log', 'cbrt' and 'eq_hist' are also supported and correspond
        to equivalent matplotlib norms. Default is the linear norm.
    cmap : str or list or :class:`matplotlib.cm.Colormap`, optional
        For scalar aggregates, a matplotlib colormap name or instance.
        Alternatively, an iterable of colors can be passed and will be converted
        to a colormap. For a single-color, transparency-based colormap, see
        :func:`alpha_colormap`.
    alpha : float
        For scalar aggregates, the alpha blending value, between 0
        (transparent) and 1 (opaque).
    vmin, vmax : float, optional
        For scalar aggregates, the data range that the colormap covers.
        If vmin or vmax is None (default), the colormap autoscales to the
        range of data in the area displayed, unless the corresponding value is
        already set in the norm.
    color_key : dict or iterable, optional
        For categorical aggregates, the colors to use for blending categories.
        See `tf.shade`.
    alpha_range : pair of int, optional
        For categorical aggregates, the minimum and maximum alpha values in
        [0, 255] to use to indicate data values of non-empty pixels. The
        default range is (40, 255).
    color_baseline : float, optional
        For categorical aggregates, the baseline for calculating how
        categorical data mixes to determine the color of a pixel. See
        `tf.shade` for more information.

    Other Parameters
    ----------------
    ax : `matplotlib.Axes`, optional
        Axes to draw into. If *None*, create a new figure or use ``fignum`` to
        draw into an existing figure.
    fignum : None or int or False, optional
        If *None* and ``ax`` is *None*, create a new figure window with
        automatic numbering.
        If a nonzero integer and ``ax`` is *None*, draw into the figure with
        the given number (create it if it does not exist).
        If 0, use the current axes (or create one if it does not exist).
    aspect : {'equal', 'auto'} or float, default: ``rcParams["image.aspect"]``
        The aspect ratio of the axes.
    **kwargs
        All other kwargs are passed to the artist.

    Returns
    -------
    :class:`ScalarDSArtist` or :class:`CategoricalDSArtist`

    Notes
    -----
    If the aggregation is scalar/single-category (i.e. generates a 2D scalar
    mappable), the artist can be used to make a colorbar. See example.

    If the aggregation is multi-category (i.e. generates a 3D array with
    two or more components that get composited to form an image), you can use
    the :meth:`CategoricalDSArtist.get_legend_elements` method to obtain patch
    handles that can be used to make a legend. See example.

    Examples
    --------
    Generate two Gaussian point clouds and (1) plot the density as a
    quantitative map and (2) color the points by category.

    .. plot::
        :context: close-figs

        >>> import pandas as pd
        >>> import datashader as ds
        >>> import matplotlib.pyplot as plt
        >>> from datashader.mpl_ext import dsshow
        >>> n = 10000
        >>> df = pd.DataFrame({
        ...     'x': np.r_[np.random.randn(n) - 1, np.random.randn(n) + 1],
        ...     'y': np.r_[np.random.randn(n), np.random.randn(n)],
        ...     'c': pd.Categorical(np.r_[['cloud 1'] * n, ['cloud 2'] * n])
        ... })
        >>> da1 = dsshow(
        ...     df,
        ...     ds.Point('x', 'y'),
        ...     aspect='equal'
        ... )
        >>> plt.colorbar(da1);  # doctest: +SKIP
        >>> da2 = dsshow(
        ...     df,
        ...     ds.Point('x', 'y'),
        ...     ds.count_cat('c'),
        ...     aspect='equal'
        ... )
        >>> plt.legend(handles=da2.get_legend_elements());  # doctest: +SKIP

    """
    import matplotlib.pyplot as plt
    if fignum == 0:
        ax = plt.gca()
    elif ax is None:
        fig = plt.figure(fignum)
        ax = fig.add_axes([0.15, 0.09, 0.775, 0.775])
    if isinstance(aggregator, reductions.by):
        artist = CategoricalDSArtist(ax, df, glyph, aggregator, agg_hook, shade_hook, plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range, width_scale=width_scale, height_scale=height_scale, color_key=color_key, alpha_range=alpha_range, color_baseline=color_baseline, **kwargs)
    else:
        if cmap is not None:
            if isinstance(cmap, list):
                cmap = mpl.colors.LinearSegmentedColormap.from_list('_datashader', cmap)
        if norm is None:
            norm = mpl.colors.Normalize()
        elif isinstance(norm, str):
            if norm == 'linear':
                norm = mpl.colors.Normalize()
            elif norm == 'log':
                norm = mpl.colors.LogNorm()
            elif norm == 'cbrt':
                norm = mpl.colors.PowerNorm(1 / 3)
            elif norm == 'eq_hist':
                norm = EqHistNormalize()
        if not isinstance(norm, mpl.colors.Normalize):
            raise ValueError("`norm` must be one of 'linear', 'log', 'cbrt', 'eq_hist', or a matplotlib norm instance.")
        if vmin is not None:
            norm.vmin = vmin
        if vmax is not None:
            norm.vmax = vmax
        artist = ScalarDSArtist(ax, df, glyph, aggregator, agg_hook, shade_hook, plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range, width_scale=width_scale, height_scale=height_scale, norm=norm, cmap=cmap, alpha=alpha, **kwargs)
    ax.add_artist(artist)
    if aspect is None:
        aspect = plt.rcParams['image.aspect']
    ax.set_aspect(aspect)
    return artist