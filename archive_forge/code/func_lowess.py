import pandas as pd
import numpy as np
def lowess(trendline_options, x_raw, x, y, x_label, y_label, non_missing):
    """LOcally WEighted Scatterplot Smoothing (LOWESS) trendline function

    Requires `statsmodels` to be installed.

    Valid keys for the `trendline_options` dict are:

    - `frac` (`float`, default `0.6666666`): the `frac` parameter from the
    `statsmodels.api.nonparametric.lowess` function
    """
    valid_options = ['frac']
    for k in trendline_options.keys():
        if k not in valid_options:
            raise ValueError("LOWESS trendline_options keys must be one of [%s] but got '%s'" % (', '.join(valid_options), k))
    import statsmodels.api as sm
    frac = trendline_options.get('frac', 0.6666666)
    y_out = sm.nonparametric.lowess(y, x, missing='drop', frac=frac)[:, 1]
    hover_header = '<b>LOWESS trendline</b><br><br>'
    return (y_out, hover_header, None)