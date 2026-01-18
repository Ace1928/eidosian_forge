from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
def panzoom(marks):
    """Helper function for panning and zooming over a set of marks.

    Creates and returns a panzoom interaction with the 'x' and 'y' dimension
    scales of the specified marks.
    """
    return PanZoom(scales={'x': sum([mark._get_dimension_scales('x', preserve_domain=True) for mark in marks], []), 'y': sum([mark._get_dimension_scales('y', preserve_domain=True) for mark in marks], [])})