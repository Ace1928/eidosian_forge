from geopandas.tools._show_versions import (
def test_get_c_info():
    C_info = _get_C_info()
    assert 'GEOS' in C_info
    assert 'GEOS lib' in C_info
    assert 'GDAL' in C_info
    assert 'GDAL data dir' in C_info
    assert 'PROJ' in C_info
    assert 'PROJ data dir' in C_info