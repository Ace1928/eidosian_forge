import plotly.graph_objs as go
import plotly.io as pio
from collections import namedtuple, OrderedDict
from ._special_inputs import IdentityMap, Constant, Range
from .trendline_functions import ols, lowess, rolling, expanding, ewm
from _plotly_utils.basevalidators import ColorscaleValidator
from plotly.colors import qualitative, sequential
import math
from packaging import version
import pandas as pd
import numpy as np
from plotly._subplots import (
def configure_animation_controls(args, constructor, fig):

    def frame_args(duration):
        return {'frame': {'duration': duration, 'redraw': constructor != go.Scatter}, 'mode': 'immediate', 'fromcurrent': True, 'transition': {'duration': duration, 'easing': 'linear'}}
    if 'animation_frame' in args and args['animation_frame'] and (len(fig.frames) > 1):
        fig.layout.updatemenus = [{'buttons': [{'args': [None, frame_args(500)], 'label': '&#9654;', 'method': 'animate'}, {'args': [[None], frame_args(0)], 'label': '&#9724;', 'method': 'animate'}], 'direction': 'left', 'pad': {'r': 10, 't': 70}, 'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'}]
        fig.layout.sliders = [{'active': 0, 'yanchor': 'top', 'xanchor': 'left', 'currentvalue': {'prefix': get_label(args, args['animation_frame']) + '='}, 'pad': {'b': 10, 't': 60}, 'len': 0.9, 'x': 0.1, 'y': 0, 'steps': [{'args': [[f.name], frame_args(0)], 'label': f.name, 'method': 'animate'} for f in fig.frames]}]