from django.contrib.gis import gdal
@property
def geographic(self):
    """Is this Spatial Reference geographic?"""
    return self.srs.geographic