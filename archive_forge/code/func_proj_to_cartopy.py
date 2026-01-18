import sys
from collections.abc import Hashable
from functools import wraps
from packaging.version import Version
from types import FunctionType
import bokeh
import numpy as np
import pandas as pd
import param
import holoviews as hv
def proj_to_cartopy(proj):
    """
    Converts a pyproj.Proj to a cartopy.crs.Projection

    (Code copied from https://github.com/fmaussion/salem)

    Parameters
    ----------
    proj: pyproj.Proj
        the projection to convert
    Returns
    -------
    a cartopy.crs.Projection object
    """
    import cartopy.crs as ccrs
    try:
        from osgeo import osr
        has_gdal = True
    except ImportError:
        has_gdal = False
    input_proj = proj
    proj = check_crs(input_proj)
    if proj is None:
        raise ValueError(f'Invalid proj projection {input_proj!r}')
    srs = proj.srs
    if has_gdal:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning, message='Neither osr\\.UseExceptions\\(\\) nor osr\\.DontUseExceptions\\(\\) has been explicitly called\\. In GDAL 4\\.0, exceptions will be enabled by default')
            s1 = osr.SpatialReference()
            s1.ImportFromProj4(proj.srs)
            if s1.ExportToProj4():
                srs = s1.ExportToProj4()
    km_proj = {'lon_0': 'central_longitude', 'lat_0': 'central_latitude', 'x_0': 'false_easting', 'y_0': 'false_northing', 'lat_ts': 'latitude_true_scale', 'o_lon_p': 'central_rotated_longitude', 'o_lat_p': 'pole_latitude', 'k': 'scale_factor', 'zone': 'zone'}
    km_globe = {'a': 'semimajor_axis', 'b': 'semiminor_axis'}
    km_std = {'lat_1': 'lat_1', 'lat_2': 'lat_2'}
    kw_proj = {}
    kw_globe = {}
    kw_std = {}
    for s in srs.split('+'):
        s = s.split('=')
        if len(s) != 2:
            continue
        k = s[0].strip()
        v = s[1].strip()
        try:
            v = float(v)
        except:
            pass
        if k == 'proj':
            if v == 'longlat':
                cl = ccrs.PlateCarree
            elif v == 'tmerc':
                cl = ccrs.TransverseMercator
                kw_proj['approx'] = True
            elif v == 'lcc':
                cl = ccrs.LambertConformal
            elif v == 'merc':
                cl = ccrs.Mercator
            elif v == 'utm':
                cl = ccrs.UTM
            elif v == 'stere':
                cl = ccrs.Stereographic
            elif v == 'ob_tran':
                cl = ccrs.RotatedPole
            else:
                raise NotImplementedError(f'Unknown projection {v}')
        if k in km_proj:
            if k == 'zone':
                v = int(v)
            kw_proj[km_proj[k]] = v
        if k in km_globe:
            kw_globe[km_globe[k]] = v
        if k in km_std:
            kw_std[km_std[k]] = v
    globe = None
    if kw_globe:
        globe = ccrs.Globe(ellipse='sphere', **kw_globe)
    if kw_std:
        kw_proj['standard_parallels'] = (kw_std['lat_1'], kw_std['lat_2'])
    if cl.__name__ == 'Mercator':
        kw_proj.pop('false_easting', None)
        kw_proj.pop('false_northing', None)
        if 'scale_factor' in kw_proj:
            kw_proj.pop('latitude_true_scale', None)
    elif cl.__name__ == 'Stereographic':
        kw_proj.pop('scale_factor', None)
        if 'latitude_true_scale' in kw_proj:
            kw_proj['true_scale_latitude'] = kw_proj['latitude_true_scale']
            kw_proj.pop('latitude_true_scale', None)
    elif cl.__name__ == 'RotatedPole':
        if 'central_longitude' in kw_proj:
            kw_proj['pole_longitude'] = kw_proj['central_longitude'] - 180
            kw_proj.pop('central_longitude', None)
    else:
        kw_proj.pop('latitude_true_scale', None)
    try:
        return cl(globe=globe, **kw_proj)
    except TypeError:
        del kw_proj['approx']
    return cl(globe=globe, **kw_proj)