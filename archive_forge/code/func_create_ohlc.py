from plotly import exceptions
from plotly.graph_objs import graph_objs
from plotly.figure_factory import utils
def create_ohlc(open, high, low, close, dates=None, direction='both', **kwargs):
    """
    **deprecated**, use instead the plotly.graph_objects trace
    :class:`plotly.graph_objects.Ohlc`

    :param (list) open: opening values
    :param (list) high: high values
    :param (list) low: low values
    :param (list) close: closing
    :param (list) dates: list of datetime objects. Default: None
    :param (string) direction: direction can be 'increasing', 'decreasing',
        or 'both'. When the direction is 'increasing', the returned figure
        consists of all units where the close value is greater than the
        corresponding open value, and when the direction is 'decreasing',
        the returned figure consists of all units where the close value is
        less than or equal to the corresponding open value. When the
        direction is 'both', both increasing and decreasing units are
        returned. Default: 'both'
    :param kwargs: kwargs passed through plotly.graph_objs.Scatter.
        These kwargs describe other attributes about the ohlc Scatter trace
        such as the color or the legend name. For more information on valid
        kwargs call help(plotly.graph_objs.Scatter)

    :rtype (dict): returns a representation of an ohlc chart figure.

    Example 1: Simple OHLC chart from a Pandas DataFrame

    >>> from plotly.figure_factory import create_ohlc
    >>> from datetime import datetime

    >>> import pandas as pd
    >>> df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
    >>> fig = create_ohlc(df['AAPL.Open'], df['AAPL.High'], df['AAPL.Low'], df['AAPL.Close'], dates=df.index)
    >>> fig.show()
    """
    if dates is not None:
        utils.validate_equal_length(open, high, low, close, dates)
    else:
        utils.validate_equal_length(open, high, low, close)
    validate_ohlc(open, high, low, close, direction, **kwargs)
    if direction == 'increasing':
        ohlc_incr = make_increasing_ohlc(open, high, low, close, dates, **kwargs)
        data = [ohlc_incr]
    elif direction == 'decreasing':
        ohlc_decr = make_decreasing_ohlc(open, high, low, close, dates, **kwargs)
        data = [ohlc_decr]
    else:
        ohlc_incr = make_increasing_ohlc(open, high, low, close, dates, **kwargs)
        ohlc_decr = make_decreasing_ohlc(open, high, low, close, dates, **kwargs)
        data = [ohlc_incr, ohlc_decr]
    layout = graph_objs.Layout(xaxis=dict(zeroline=False), hovermode='closest')
    return graph_objs.Figure(data=data, layout=layout)