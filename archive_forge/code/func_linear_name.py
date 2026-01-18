from django.contrib.gis import gdal
@property
def linear_name(self):
    """Return the linear units name."""
    return self.srs.linear_name