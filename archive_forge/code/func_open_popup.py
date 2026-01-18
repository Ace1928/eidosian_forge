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
def open_popup(self, location=None):
    """Open the popup on the bound map.

        Parameters
        ----------
        location: list, default to the internal location
            The location to open the popup at.
        """
    if location is not None:
        self.location = location
    self.send({'msg': 'open', 'location': self.location if location is None else location})