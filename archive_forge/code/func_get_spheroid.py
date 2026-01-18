from django.contrib.gis import gdal
@classmethod
def get_spheroid(cls, wkt, string=True):
    """
        Class method used by GeometryField on initialization to
        retrieve the `SPHEROID[..]` parameters from the given WKT.
        """
    srs = gdal.SpatialReference(wkt)
    sphere_params = srs.ellipsoid
    sphere_name = srs['spheroid']
    if not string:
        return (sphere_name, sphere_params)
    else:
        if len(sphere_params) == 3:
            radius, flattening = (sphere_params[0], sphere_params[2])
        else:
            radius, flattening = sphere_params
        return 'SPHEROID["%s",%s,%s]' % (sphere_name, radius, flattening)