import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def color_parser(colors, function):
    """
    Takes color(s) and a function and applies the function on the color(s)

    In particular, this function identifies whether the given color object
    is an iterable or not and applies the given color-parsing function to
    the color or iterable of colors. If given an iterable, it will only be
    able to work with it if all items in the iterable are of the same type
    - rgb string, hex string or tuple
    """
    if isinstance(colors, str):
        return function(colors)
    if isinstance(colors, tuple) and isinstance(colors[0], Number):
        return function(colors)
    if hasattr(colors, '__iter__'):
        if isinstance(colors, tuple):
            new_color_tuple = tuple((function(item) for item in colors))
            return new_color_tuple
        else:
            new_color_list = [function(item) for item in colors]
            return new_color_list