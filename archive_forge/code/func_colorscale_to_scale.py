import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def colorscale_to_scale(colorscale):
    """
    Extracts the interpolation scale values from colorscale as a list
    """
    scale_list = []
    for item in colorscale:
        scale_list.append(item[0])
    return scale_list