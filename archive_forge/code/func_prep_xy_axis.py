import math
import warnings
import matplotlib.dates
def prep_xy_axis(ax, props, x_bounds, y_bounds):
    xaxis = dict(type=props['axes'][0]['scale'], range=list(props['xlim']), showgrid=props['axes'][0]['grid']['gridOn'], domain=convert_x_domain(props['bounds'], x_bounds), side=props['axes'][0]['position'], tickfont=dict(size=props['axes'][0]['fontsize']))
    xaxis.update(prep_ticks(ax, 0, 'x', props))
    yaxis = dict(type=props['axes'][1]['scale'], range=list(props['ylim']), showgrid=props['axes'][1]['grid']['gridOn'], domain=convert_y_domain(props['bounds'], y_bounds), side=props['axes'][1]['position'], tickfont=dict(size=props['axes'][1]['fontsize']))
    yaxis.update(prep_ticks(ax, 1, 'y', props))
    return (xaxis, yaxis)