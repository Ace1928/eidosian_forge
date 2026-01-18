import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def convert_dict_colors_to_same_type(colors_dict, colortype='rgb'):
    """
    Converts a colors in a dictionary of colors to the specified color type

    :param (dict) colors_dict: a dictionary whose values are single colors
    """
    for key in colors_dict:
        if '#' in colors_dict[key]:
            colors_dict[key] = color_parser(colors_dict[key], hex_to_rgb)
            colors_dict[key] = color_parser(colors_dict[key], label_rgb)
        elif isinstance(colors_dict[key], tuple):
            colors_dict[key] = color_parser(colors_dict[key], convert_to_RGB_255)
            colors_dict[key] = color_parser(colors_dict[key], label_rgb)
    if colortype == 'rgb':
        return colors_dict
    elif colortype == 'tuple':
        for key in colors_dict:
            colors_dict[key] = color_parser(colors_dict[key], unlabel_rgb)
            colors_dict[key] = color_parser(colors_dict[key], unconvert_from_RGB_255)
        return colors_dict
    else:
        raise exceptions.PlotlyError('You must select either rgb or tuple for your colortype variable.')