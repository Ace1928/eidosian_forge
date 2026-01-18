import re
import numpy as np
from pandas import DataFrame
from ...rcparams import rcParams
def create_layout(ax, force_layout=False):
    """Transform bokeh array of figures to layout."""
    ax = np.atleast_2d(ax)
    subplot_order = rcParams['plot.bokeh.layout.order']
    if force_layout:
        from bokeh.layouts import gridplot as layout
        ax = ax.tolist()
        layout_args = {'sizing_mode': rcParams['plot.bokeh.layout.sizing_mode'], 'toolbar_location': rcParams['plot.bokeh.layout.toolbar_location']}
    elif any((item in subplot_order for item in ('row', 'column'))):
        match = re.match('(\\d*)(row|column)', subplot_order)
        n = int(match.group(1)) if match.group(1) is not None else 1
        subplot_order = match.group(2)
        ax = [item for item in ax.ravel().tolist() if item is not None]
        layout_args = {'sizing_mode': rcParams['plot.bokeh.layout.sizing_mode']}
        if subplot_order == 'row' and n == 1:
            from bokeh.layouts import row as layout
        elif subplot_order == 'column' and n == 1:
            from bokeh.layouts import column as layout
        else:
            from bokeh.layouts import layout
        if n != 1:
            ax = np.array(ax + [None for _ in range(int(np.ceil(len(ax) / n)) - len(ax))])
            ax = ax.reshape(n, -1) if subplot_order == 'row' else ax.reshape(-1, n)
            ax = ax.tolist()
    else:
        if subplot_order in ('square', 'square_trimmed'):
            ax = [item for item in ax.ravel().tolist() if item is not None]
            n = int(np.ceil(len(ax) ** 0.5))
            ax = ax + [None for _ in range(n ** 2 - len(ax))]
            ax = np.array(ax).reshape(n, n)
        ax = ax.tolist()
        if subplot_order == 'square_trimmed' and any((all((item is None for item in row)) for row in ax)):
            from bokeh.layouts import layout
            ax = [row for row in ax if any((item is not None for item in row))]
            layout_args = {'sizing_mode': rcParams['plot.bokeh.layout.sizing_mode']}
        else:
            from bokeh.layouts import gridplot as layout
            layout_args = {'sizing_mode': rcParams['plot.bokeh.layout.sizing_mode'], 'toolbar_location': rcParams['plot.bokeh.layout.toolbar_location']}
    if layout_args.get('sizing_mode', '') == 'fixed':
        layout_args.pop('sizing_mode')
    return layout(ax, **layout_args)