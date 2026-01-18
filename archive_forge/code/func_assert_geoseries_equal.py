import warnings
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from geopandas import _vectorized
def assert_geoseries_equal(left, right, check_dtype=True, check_index_type=False, check_series_type=True, check_less_precise=False, check_geom_type=False, check_crs=True, normalize=False):
    """
    Test util for checking that two GeoSeries are equal.

    Parameters
    ----------
    left, right : two GeoSeries
    check_dtype : bool, default False
        If True, check geo dtype [only included so it's a drop-in replacement
        for assert_series_equal].
    check_index_type : bool, default False
        Check that index types are equal.
    check_series_type : bool, default True
        Check that both are same type (*and* are GeoSeries). If False,
        will attempt to convert both into GeoSeries.
    check_less_precise : bool, default False
        If True, use geom_equals_exact with relative error of 0.5e-6.
        If False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_crs: bool, default True
        If `check_series_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_equals_exact`` and requires exact coordinate order.
    """
    assert len(left) == len(right), '%d != %d' % (len(left), len(right))
    if check_dtype:
        msg = 'dtype should be a GeometryDtype, got {0}'
        assert isinstance(left.dtype, GeometryDtype), msg.format(left.dtype)
        assert isinstance(right.dtype, GeometryDtype), msg.format(left.dtype)
    if check_index_type:
        assert isinstance(left.index, type(right.index))
    if check_series_type:
        assert isinstance(left, GeoSeries)
        assert isinstance(left, type(right))
        if check_crs:
            assert left.crs == right.crs
    else:
        if not isinstance(left, GeoSeries):
            left = GeoSeries(left)
        if not isinstance(right, GeoSeries):
            right = GeoSeries(right, index=left.index)
    assert left.index.equals(right.index), 'index: %s != %s' % (left.index, right.index)
    if check_geom_type:
        assert (left.geom_type == right.geom_type).all(), 'type: %s != %s' % (left.geom_type, right.geom_type)
    if normalize:
        left = GeoSeries(_vectorized.normalize(left.array._data))
        right = GeoSeries(_vectorized.normalize(right.array._data))
    if not check_crs:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'CRS mismatch', UserWarning)
            _check_equality(left, right, check_less_precise)
    else:
        _check_equality(left, right, check_less_precise)