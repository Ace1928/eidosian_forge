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
@property
def direction(self):
    """The unit vector direction towards the light source."""
    az = np.radians(90 - self.azdeg)
    alt = np.radians(self.altdeg)
    return np.array([np.cos(az) * np.cos(alt), np.sin(az) * np.cos(alt), np.sin(alt)])