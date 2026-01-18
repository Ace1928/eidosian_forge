import dask.dataframe as dd
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray
import numpy as np
from numpy import nan
import pytest
@pytest.mark.skipif(not geodatasets, reason='geodatasets not installed')
@pytest.mark.skipif(not spatialpandas, reason='spatialpandas not installed')
@pytest.mark.parametrize('npartitions', [1, 2, 5])
@pytest.mark.parametrize('geom_type, explode, use_boundary', [('multipolygon', False, False), ('polygon', True, False), ('multilinestring', False, True), ('linestring', True, True)])
def test_lines_spatialpandas(geom_type, explode, use_boundary, npartitions):
    df = geopandas.read_file(geodatasets.get_path('nybb'))
    df['col'] = np.arange(len(df))
    geometry = 'boundary' if use_boundary else 'geometry'
    if explode:
        df = df.explode(index_parts=False)
    if use_boundary:
        df['boundary'] = df.boundary
    unique_geom_type = df[geometry].geom_type.unique()
    assert len(unique_geom_type) == 1 and unique_geom_type[0].lower() == geom_type
    df = spatialpandas.GeoDataFrame(df)
    if npartitions > 0:
        df = dd.from_pandas(df, npartitions=npartitions)
        assert df.npartitions == npartitions
    canvas = ds.Canvas(plot_height=20, plot_width=20)
    agg = canvas.line(source=df, geometry=geometry, agg=ds.max('col'))
    assert_eq_ndarray(agg.data, nybb_lines_sol)