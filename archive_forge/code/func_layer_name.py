from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.field import Field
from django.contrib.gis.gdal.geometries import OGRGeometry, OGRGeomType
from django.contrib.gis.gdal.prototypes import ds as capi
from django.contrib.gis.gdal.prototypes import geom as geom_api
from django.utils.encoding import force_bytes, force_str
@property
def layer_name(self):
    """Return the name of the layer for the feature."""
    name = capi.get_feat_name(self._layer._ldefn)
    return force_str(name, self.encoding, strings_only=True)