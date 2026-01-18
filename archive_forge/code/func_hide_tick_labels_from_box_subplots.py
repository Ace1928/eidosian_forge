from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def hide_tick_labels_from_box_subplots(fig):
    """
    Hides tick labels for box plots in scatterplotmatrix subplots.
    """
    boxplot_xaxes = []
    for trace in fig['data']:
        if trace['type'] == 'box':
            boxplot_xaxes.append('xaxis{}'.format(trace['xaxis'][1:]))
    for xaxis in boxplot_xaxes:
        fig['layout'][xaxis]['showticklabels'] = False