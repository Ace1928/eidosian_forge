import itertools
import io
import base64
import numpy as np
import warnings
import matplotlib
from matplotlib.colors import colorConverter
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib import ticker
def get_dasharray(obj):
    """Get an SVG dash array for the given matplotlib linestyle

    Parameters
    ----------
    obj : matplotlib object
        The matplotlib line or path object, which must have a get_linestyle()
        method which returns a valid matplotlib line code

    Returns
    -------
    dasharray : string
        The HTML/SVG dasharray code associated with the object.
    """
    if obj.__dict__.get('_dashSeq', None) is not None:
        return ','.join(map(str, obj._dashSeq))
    else:
        ls = obj.get_linestyle()
        dasharray = LINESTYLES.get(ls, 'not found')
        if dasharray == 'not found':
            warnings.warn("line style '{0}' not understood: defaulting to solid line.".format(ls))
            dasharray = LINESTYLES['solid']
        return dasharray