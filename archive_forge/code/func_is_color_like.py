import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def is_color_like(c):
    """Return whether *c* can be interpreted as an RGB(A) color."""
    if _is_nth_color(c):
        return True
    try:
        to_rgba(c)
    except ValueError:
        return False
    else:
        return True