from numbers import Number
from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
from plotly.subplots import make_subplots
def make_half_violin(x, y, fillcolor='#1f77b4', linecolor='rgb(0, 0, 0)'):
    """
    Produces a sideways probability distribution fig violin plot.
    """
    text = ['(pdf(y), y)=(' + '{:0.2f}'.format(x[i]) + ', ' + '{:0.2f}'.format(y[i]) + ')' for i in range(len(x))]
    return graph_objs.Scatter(x=x, y=y, mode='lines', name='', text=text, fill='tonextx', fillcolor=fillcolor, line=graph_objs.scatter.Line(width=0.5, color=linecolor, shape='spline'), hoverinfo='text', opacity=0.5)