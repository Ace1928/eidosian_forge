from __future__ import annotations
import logging # isort:skip
import sys
import types
import xyzservices
from bokeh.core.enums import enumeration
from .util.deprecation import deprecated
def deprecated_vendors():
    deprecated((3, 0, 0), 'tile_providers module', 'add_tile directly')
    return enumeration('CARTODBPOSITRON', 'CARTODBPOSITRON_RETINA', 'STAMEN_TERRAIN', 'STAMEN_TERRAIN_RETINA', 'STAMEN_TONER', 'STAMEN_TONER_BACKGROUND', 'STAMEN_TONER_LABELS', 'OSM', 'ESRI_IMAGERY', case_sensitive=True)