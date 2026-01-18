import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def iplot_mpl(mpl_fig, resize=False, strip_style=False, verbose=False, show_link=False, link_text='Export to plot.ly', validate=True, image=None, image_filename='plot_image', image_height=600, image_width=800):
    """
    Convert a matplotlib figure to a plotly graph and plot inside an IPython
    notebook without connecting to an external server.

    To save the chart to Plotly Cloud or Plotly Enterprise, use
    `plotly.plotly.plot_mpl`.

    For more information on converting matplotlib visualizations to plotly
    graphs call `help(plotly.tools.mpl_to_plotly)`

    For more information on plotting plotly charts offline in an Ipython
    notebook call `help(plotly.offline.iplot)`

    mpl_fig -- a matplotlib.figure to convert to a plotly graph

    Keyword arguments:
    resize (default=False) -- allow plotly to choose the figure size.
    strip_style (default=False) -- allow plotly to choose style options.
    verbose (default=False) -- print message.
    show_link (default=False) -- display a link in the bottom-right corner of
                                of the chart that will export the chart to
                                Plotly Cloud or Plotly Enterprise
    link_text (default='Export to plot.ly') -- the text of export link
    validate (default=True) -- validate that all of the keys in the figure
                               are valid? omit if your version of plotly.js
                               has become outdated with your version of
                               graph_reference.json or if you need to include
                               extra, unnecessary keys in your figure.
    image (default=None |'png' |'jpeg' |'svg' |'webp') -- This parameter sets
        the format of the image to be downloaded, if we choose to download an
        image. This parameter has a default value of None indicating that no
        image should be downloaded.
    image_filename (default='plot_image') -- Sets the name of the file your
        image will be saved to. The extension should not be included.
    image_height (default=600) -- Specifies the height of the image in `px`.
    image_width (default=800) -- Specifies the width of the image in `px`.

    Example:
    ```
    from plotly.offline import init_notebook_mode, iplot_mpl
    import matplotlib.pyplot as plt

    fig = plt.figure()
    x = [10, 15, 20, 25, 30]
    y = [100, 250, 200, 150, 300]
    plt.plot(x, y, "o")

    init_notebook_mode()
    iplot_mpl(fig)
    # and if you want to download an image of the figure as well
    iplot_mpl(fig, image='jpeg')
    ```
    """
    plotly_plot = tools.mpl_to_plotly(mpl_fig, resize, strip_style, verbose)
    return iplot(plotly_plot, show_link, link_text, validate, image=image, filename=image_filename, image_height=image_height, image_width=image_width)