import math
import warnings
import matplotlib.dates
def convert_rgba_array(color_list):
    clean_color_list = list()
    for c in color_list:
        clean_color_list += [dict(r=int(c[0] * 255), g=int(c[1] * 255), b=int(c[2] * 255), a=c[3])]
    plotly_colors = list()
    for rgba in clean_color_list:
        plotly_colors += ['rgba({r},{g},{b},{a})'.format(**rgba)]
    if len(plotly_colors) == 1:
        return plotly_colors[0]
    else:
        return plotly_colors