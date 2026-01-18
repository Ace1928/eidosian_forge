from django.contrib.gis.gdal.base import GDALBase
from django.contrib.gis.gdal.error import GDALException
from django.contrib.gis.gdal.field import Field
from django.contrib.gis.gdal.geometries import OGRGeometry, OGRGeomType
from django.contrib.gis.gdal.prototypes import ds as capi
from django.contrib.gis.gdal.prototypes import geom as geom_api
from django.utils.encoding import force_bytes, force_str
@property
def geom(self):
    """Return the OGR Geometry for this Feature."""
    geom_ptr = capi.get_feat_geom_ref(self.ptr)
    return OGRGeometry(geom_api.clone_geom(geom_ptr))