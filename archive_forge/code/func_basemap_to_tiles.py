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
def basemap_to_tiles(basemap, day=yesterday, **kwargs):
    """Turn a basemap into a TileLayer object.

    Parameters
    ----------
    basemap : class:`xyzservices.lib.TileProvider` or Dict
        Basemap description coming from ipyleaflet.basemaps.
    day: string
        If relevant for the chosen basemap, you can specify the day for
        the tiles in the "%Y-%m-%d" format. Defaults to yesterday's date.
    kwargs: key-word arguments
        Extra key-word arguments to pass to the TileLayer constructor.
    """
    if isinstance(basemap, xyzservices.lib.TileProvider):
        url = basemap.build_url(time=day)
    elif isinstance(basemap, dict):
        url = basemap.get('url', '')
    else:
        raise ValueError('Invalid basemap type')
    return TileLayer(url=url, max_zoom=basemap.get('max_zoom', 18), min_zoom=basemap.get('min_zoom', 1), attribution=basemap.get('html_attribution', '') or basemap.get('attribution', ''), name=basemap.get('name', ''), **kwargs)