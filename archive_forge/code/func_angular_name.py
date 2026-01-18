from django.contrib.gis import gdal
@property
def angular_name(self):
    """Return the name of the angular units."""
    return self.srs.angular_name