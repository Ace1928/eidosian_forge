from django.contrib.gis.geos import prototypes as capi
from django.contrib.gis.geos.geometry import GEOSGeometry
from django.contrib.gis.geos.libgeos import GEOM_PTR
from django.contrib.gis.geos.linestring import LinearRing
@property
def num_interior_rings(self):
    """Return the number of interior rings."""
    return capi.get_nrings(self.ptr)