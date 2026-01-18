import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def make_colorscale(colors, scale=None):
    """
    Makes a colorscale from a list of colors and a scale

    Takes a list of colors and scales and constructs a colorscale based
    on the colors in sequential order. If 'scale' is left empty, a linear-
    interpolated colorscale will be generated. If 'scale' is a specificed
    list, it must be the same legnth as colors and must contain all floats
    For documentation regarding to the form of the output, see
    https://plot.ly/python/reference/#mesh3d-colorscale

    :param (list) colors: a list of single colors
    """
    colorscale = []
    if len(colors) < 2:
        raise exceptions.PlotlyError('You must input a list of colors that has at least two colors.')
    if scale is None:
        scale_incr = 1.0 / (len(colors) - 1)
        return [[i * scale_incr, color] for i, color in enumerate(colors)]
    else:
        if len(colors) != len(scale):
            raise exceptions.PlotlyError('The length of colors and scale must be the same.')
        validate_scale_values(scale)
        colorscale = [list(tup) for tup in zip(scale, colors)]
        return colorscale