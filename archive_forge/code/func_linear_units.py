from django.contrib.gis import gdal
@property
def linear_units(self):
    """Return the linear units."""
    return self.srs.linear_units