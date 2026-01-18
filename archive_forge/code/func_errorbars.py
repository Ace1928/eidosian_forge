import itertools
from collections import defaultdict
import param
from ..converter import HoloViewsConverter
from ..util import is_list_like, process_dynamic_args
def errorbars(self, x=None, y=None, yerr1=None, yerr2=None, **kwds):
    """
        `errorbars` provide a visual indicator for the variability of the plotted data on a graph.
        They are usually overlaid with other plots such as `scatter` , `line` or `bar` plots to
        indicate the variability.

        Reference: https://hvplot.holoviz.org/reference/tabular/errorbars.html

        Parameters
        ----------
        x : string, optional
            Field name to draw the x-position from. If not specified, the index is
            used. Can refer to continuous and categorical data.
        y : string, optional
            Field name to draw the y-position from
        yerr1 : string, optional
            Field name to draw symmetric / negative errors from
        yerr2 : string, optional
            Field name to draw positive errors from
        **kwds : optional
            Additional keywords arguments are documented in `hvplot.help('errorbars')`.

        Returns
        -------
        A Holoviews object. You can `print` the object to study its composition and run

        .. code-block::

            import holoviews as hv
            hv.help(the_holoviews_object)

        to learn more about its parameters and options.

        Example
        -------

        .. code-block::

            import hvplot.pandas
            import pandas as pd

            df = pd.DataFrame(
                {
                    "actual": [100, 150, 125, 140, 145, 135, 123],
                    "forecast": [90, 160, 125, 150, 141, 141, 120],
                    "numerical": [1.1, 1.9, 3.2, 3.8, 4.3, 5.0, 5.5],
                    "date": pd.date_range("2022-01-03", "2022-01-09"),
                    "string": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                },
            )
            df["min"] = df[["actual", "forecast"]].min(axis=1)
            df["max"] = df[["actual", "forecast"]].max(axis=1)
            df["mean"] = df[["actual", "forecast"]].mean(axis=1)
            df["yerr2"] = df["max"] - df["mean"]
            df["yerr1"] = df["mean"] - df["min"]

            errorbars = df.hvplot.errorbars(
                x="numerical",
                y="mean",
                yerr1="yerr1",
                yerr2="yerr2",
                legend="bottom",
                height=500,
                alpha=0.5,
                line_width=2,
            )
            errorbars

        Normally you would overlay the `errorbars` on for example a `scatter` plot.

        .. code-block::

            mean = df.hvplot.scatter(x="numerical", y=["mean"], color=["#55a194"], size=50)
            errorbars * mean

        References
        ----------

        - Bokeh: https://docs.bokeh.org/en/latest/docs/user_guide/annotations.html#whiskers
        - HoloViews: https://holoviews.org/reference/elements/bokeh/ErrorBars.html
        - Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
        - Pandas: https://pandas.pydata.org/docs/user_guide/visualization.html#visualization-errorbars
        - Plotly: https://plotly.com/python/error-bars/
        - Wikipedia: https://en.wikipedia.org/wiki/Error_bar
        """
    return self(x, y, kind='errorbars', yerr1=yerr1, yerr2=yerr2, **kwds)