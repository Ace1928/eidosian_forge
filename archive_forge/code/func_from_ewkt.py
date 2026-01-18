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
@staticmethod
def from_ewkt(ewkt):
    ewkt = force_bytes(ewkt)
    srid = None
    parts = ewkt.split(b';', 1)
    if len(parts) == 2:
        srid_part, wkt = parts
        match = re.match(b'SRID=(?P<srid>\\-?\\d+)', srid_part)
        if not match:
            raise ValueError('EWKT has invalid SRID part.')
        srid = int(match['srid'])
    else:
        wkt = ewkt
    if not wkt:
        raise ValueError('Expected WKT but got an empty string.')
    return GEOSGeometry(GEOSGeometry._from_wkt(wkt), srid=srid)