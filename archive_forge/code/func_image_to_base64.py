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
def image_to_base64(image):
    """
    Convert a matplotlib image to a base64 png representation

    Parameters
    ----------
    image : matplotlib image object
        The image to be converted.

    Returns
    -------
    image_base64 : string
        The UTF8-encoded base64 string representation of the png image.
    """
    ax = image.axes
    binary_buffer = io.BytesIO()
    lim = ax.axis()
    ax.axis(image.get_extent())
    image.write_png(binary_buffer)
    ax.axis(lim)
    binary_buffer.seek(0)
    return base64.b64encode(binary_buffer.read()).decode('utf-8')