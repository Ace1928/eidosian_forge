import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def get_plotlyjs():
    """
    Return the contents of the minified plotly.js library as a string.

    This may be useful when building standalone HTML reports.

    Returns
    -------
    str
        Contents of the minified plotly.js library as a string

    Examples
    --------
    Here is an example of creating a standalone HTML report that contains
    two plotly figures, each in their own div.  The include_plotlyjs argument
    is set to False when creating the divs so that we don't include multiple
    copies of the plotly.js library in the output.  Instead, a single copy
    of plotly.js is included in a script tag in the html head element.

    >>> import plotly.graph_objs as go
    >>> from plotly.offline import plot, get_plotlyjs
    >>> fig1 = go.Figure(data=[{'type': 'bar', 'y': [1, 3, 2]}],
    ...                 layout={'height': 400})
    >>> fig2 = go.Figure(data=[{'type': 'scatter', 'y': [1, 3, 2]}],
    ...                  layout={'height': 400})
    >>> div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    >>> div2 = plot(fig2, output_type='div', include_plotlyjs=False)

    >>> html = '''
    ... <html>
    ...     <head>
    ...         <script type="text/javascript">{plotlyjs}</script>
    ...     </head>
    ...     <body>
    ...        {div1}
    ...        {div2}
    ...     </body>
    ... </html>
    ... '''.format(plotlyjs=get_plotlyjs(), div1=div1, div2=div2)

    >>> with open('multi_plot.html', 'w') as f:
    ...      f.write(html) # doctest: +SKIP
    """
    path = os.path.join('package_data', 'plotly.min.js')
    plotlyjs = pkgutil.get_data('plotly', path).decode('utf-8')
    return plotlyjs