from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def make_quartiles(q1, q3):
    """
    Makes the upper and lower quartiles for a violin plot.
    """
    return graph_objs.Scatter(x=[0, 0], y=[q1, q3], text=['lower-quartile: ' + '{:0.2f}'.format(q1), 'upper-quartile: ' + '{:0.2f}'.format(q3)], mode='lines', line=graph_objs.scatter.Line(width=4, color='rgb(0,0,0)'), hoverinfo='text')