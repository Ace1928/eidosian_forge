import re
from ctypes import addressof, byref, c_double
from django.contrib.gis import gdal
from django.contrib.gis.geometry import hex_regex, json_regex, wkt_regex
from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.base import GEOSBase
from django.contrib.gis.geos.coordseq import GEOSCoordSeq
from django.contrib.gis.geos.error import GEOSException
from django.contrib.gis.geos.libgeos import GEOM_PTR, geos_version_tuple
from django.contrib.gis.geos.mutable_list import ListMixin
from django.contrib.gis.geos.prepared import PreparedGeometry
from django.contrib.gis.geos.prototypes.io import ewkb_w, wkb_r, wkb_w, wkt_r, wkt_w
from django.utils.deconstruct import deconstructible
from django.utils.encoding import force_bytes, force_str
@property
def num_points(self):
    """Return the number points, or coordinates, in the Geometry."""
    return self.num_coords