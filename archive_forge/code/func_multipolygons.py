import numpy as np
from shapely import Geometry, GeometryType, lib
from shapely._geometry_helpers import collections_1d, simple_geometries_1d
from shapely.decorators import multithreading_enabled
from shapely.io import from_wkt
@multithreading_enabled
def multipolygons(geometries, indices=None, out=None, **kwargs):
    """Create multipolygons from arrays of polygons

    Parameters
    ----------
    geometries : array_like
        An array of polygons or coordinates (see polygons).
    indices : array_like, optional
        Indices into the target array where input geometries belong. If
        provided, both geometries and indices should be 1D and have matching
        sizes. Indices should be in increasing order. Missing indices result
        in a ValueError unless ``out`` is  provided, in which case the original
        value in ``out`` is kept.
    out : ndarray, optional
        An array (with dtype object) to output the geometries into.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.
        Ignored if ``indices`` is provided.

    See also
    --------
    multipoints
    """
    typ = GeometryType.MULTIPOLYGON
    geometries = np.asarray(geometries)
    if not isinstance(geometries, Geometry) and np.issubdtype(geometries.dtype, np.number):
        geometries = polygons(geometries)
    if indices is None:
        return lib.create_collection(geometries, typ, out=out, **kwargs)
    else:
        return collections_1d(geometries, indices, typ, out=out)