import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely._ragged_array import from_ragged_array, to_ragged_array
from shapely.decorators import requires_geos
from shapely.errors import UnsupportedGEOSVersionError
@requires_geos('3.10.1')
def from_geojson(geometry, on_invalid='raise', **kwargs):
    """Creates geometries from GeoJSON representations (strings).

    If a GeoJSON is a FeatureCollection, it is read as a single geometry
    (with type GEOMETRYCOLLECTION). This may be unpacked using the ``pygeos.get_parts``.
    Properties are not read.

    The GeoJSON format is defined in `RFC 7946 <https://geojson.org/>`__.

    The following are currently unsupported:

    - Three-dimensional geometries: the third dimension is ignored.
    - Geometries having 'null' in the coordinates.

    Parameters
    ----------
    geometry : str, bytes or array_like
        The GeoJSON string or byte object(s) to convert.
    on_invalid : {"raise", "warn", "ignore"}, default "raise"
        - raise: an exception will be raised if an input GeoJSON is invalid.
        - warn: a warning will be raised and invalid input geometries will be
          returned as ``None``.
        - ignore: invalid input geometries will be returned as ``None`` without a warning.
    **kwargs
        See :ref:`NumPy ufunc docs <ufuncs.kwargs>` for other keyword arguments.

    See also
    --------
    get_parts

    Examples
    --------
    >>> from_geojson('{"type": "Point","coordinates": [1, 2]}')
    <POINT (1 2)>
    """
    if not np.isscalar(on_invalid):
        raise TypeError('on_invalid only accepts scalar values')
    invalid_handler = np.uint8(DecodingErrorOptions.get_value(on_invalid))
    geometry = np.asarray(geometry, dtype=object)
    return lib.from_geojson(geometry, invalid_handler, **kwargs)