import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def label_rgb(colors):
    """
    Takes tuple (a, b, c) and returns an rgb color 'rgb(a, b, c)'
    """
    return 'rgb(%s, %s, %s)' % (colors[0], colors[1], colors[2])