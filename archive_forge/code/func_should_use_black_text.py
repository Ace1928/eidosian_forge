import plotly.colors as clrs
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.validators.heatmap import ColorscaleValidator
def should_use_black_text(background_color):
    return background_color[0] * 0.299 + background_color[1] * 0.587 + background_color[2] * 0.114 > 186