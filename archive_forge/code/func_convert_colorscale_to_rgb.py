import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def convert_colorscale_to_rgb(colorscale):
    """
    Converts the colors in a colorscale to rgb colors

    A colorscale is an array of arrays, each with a numeric value as the
    first item and a color as the second. This function specifically is
    converting a colorscale with tuple colors (each coordinate between 0
    and 1) into a colorscale with the colors transformed into rgb colors
    """
    for color in colorscale:
        color[1] = convert_to_RGB_255(color[1])
    for color in colorscale:
        color[1] = label_rgb(color[1])
    return colorscale