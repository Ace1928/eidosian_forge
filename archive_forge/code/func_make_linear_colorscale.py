from numbers import Number
import plotly.exceptions
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
def make_linear_colorscale(colors):
    """
    Makes a list of colors into a colorscale-acceptable form

    For documentation regarding to the form of the output, see
    https://plot.ly/python/reference/#mesh3d-colorscale
    """
    scale = 1.0 / (len(colors) - 1)
    return [[i * scale, color] for i, color in enumerate(colors)]