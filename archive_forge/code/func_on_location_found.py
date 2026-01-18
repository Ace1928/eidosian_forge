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
def on_location_found(self, callback, remove=False):
    """Add a found location event listener. The callback will be called when a search result has been found.

        Parameters
        ----------
        callback : callable
            Callback function that will be called on location found event.
        remove: boolean
            Whether to remove this callback or not. Defaults to False.
        """
    self._location_found_callbacks.register_callback(callback, remove=remove)