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
@observe('attribution_control')
def observe_attribution_control(self, change):
    if change['new']:
        self.attribution_control_instance = AttributionControl(position='bottomright')
        self.add(self.attribution_control_instance)
    elif self.attribution_control_instance is not None and self.attribution_control_instance in self.controls:
        self.remove(self.attribution_control_instance)