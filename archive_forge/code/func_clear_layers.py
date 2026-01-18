import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
def clear_layers(self):
    """Remove all layers from the map.

        .. deprecated :: 0.17.0
           Use add method instead.

        """
    warnings.warn('clear_layers is deprecated, use clear instead', DeprecationWarning)
    self.layers = ()