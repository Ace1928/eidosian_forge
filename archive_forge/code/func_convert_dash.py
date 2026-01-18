import math
import warnings
import matplotlib.dates
def convert_dash(mpl_dash):
    """Convert mpl line symbol to plotly line symbol and return symbol."""
    if mpl_dash in DASH_MAP:
        return DASH_MAP[mpl_dash]
    else:
        dash_array = mpl_dash.split(',')
        if len(dash_array) < 2:
            return 'solid'
        if math.isclose(float(dash_array[1]), 0.0):
            return 'solid'
        dashpx = ','.join([x + 'px' for x in dash_array])
        if dashpx == '7.4px,3.2px':
            dashpx = 'dashed'
        elif dashpx == '12.8px,3.2px,2.0px,3.2px':
            dashpx = 'dashdot'
        elif dashpx == '2.0px,3.3px':
            dashpx = 'dotted'
        return dashpx