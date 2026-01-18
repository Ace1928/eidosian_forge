import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def find_intermediate_color(lowcolor, highcolor, intermed, colortype='tuple'):
    """
    Returns the color at a given distance between two colors

    This function takes two color tuples, where each element is between 0
    and 1, along with a value 0 < intermed < 1 and returns a color that is
    intermed-percent from lowcolor to highcolor. If colortype is set to 'rgb',
    the function will automatically convert the rgb type to a tuple, find the
    intermediate color and return it as an rgb color.
    """
    if colortype == 'rgb':
        lowcolor = unlabel_rgb(lowcolor)
        highcolor = unlabel_rgb(highcolor)
    diff_0 = float(highcolor[0] - lowcolor[0])
    diff_1 = float(highcolor[1] - lowcolor[1])
    diff_2 = float(highcolor[2] - lowcolor[2])
    inter_med_tuple = (lowcolor[0] + intermed * diff_0, lowcolor[1] + intermed * diff_1, lowcolor[2] + intermed * diff_2)
    if colortype == 'rgb':
        inter_med_rgb = label_rgb(inter_med_tuple)
        return inter_med_rgb
    return inter_med_tuple