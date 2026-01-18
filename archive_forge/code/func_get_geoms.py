from ctypes import byref, c_double
from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.envelope import Envelope, OGREnvelope
from django.contrib.gis.gdal.error import GDALException, SRSException
from django.contrib.gis.gdal.feature import Feature
from django.contrib.gis.gdal.field import OGRFieldTypes
from django.contrib.gis.gdal.geometries import OGRGeometry
from django.contrib.gis.gdal.geomtype import OGRGeomType
from django.contrib.gis.gdal.prototypes import ds as capi
from django.contrib.gis.gdal.prototypes import geom as geom_api
from django.contrib.gis.gdal.prototypes import srs as srs_api
from django.contrib.gis.gdal.srs import SpatialReference
from django.utils.encoding import force_bytes, force_str
def get_geoms(self, geos=False):
    """
        Return a list containing the OGRGeometry for every Feature in
        the Layer.
        """
    if geos:
        from django.contrib.gis.geos import GEOSGeometry
        return [GEOSGeometry(feat.geom.wkb) for feat in self]
    else:
        return [feat.geom for feat in self]