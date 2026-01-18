import gc
from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.shapereader
from cartopy.mpl import _MPL_38
from cartopy.mpl.feature_artist import FeatureArtist
import cartopy.mpl.geoaxes as cgeoaxes
import cartopy.mpl.patch
def test_contourf_transform_path_counting():
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Robinson())
    fig.canvas.draw()
    gc.collect()
    initial_cache_size = len(cgeoaxes._PATH_TRANSFORM_CACHE)
    with mock.patch('cartopy.mpl.patch.path_to_geos') as path_to_geos_counter:
        x, y, z = sample_data((30, 60))
        cs = ax.contourf(x, y, z, 5, transform=ccrs.PlateCarree())
        if not _MPL_38:
            n_geom = sum((len(c.get_paths()) for c in cs.collections))
        else:
            n_geom = len(cs.get_paths())
        del cs
        fig.canvas.draw()
    assert path_to_geos_counter.call_count == n_geom, f'The given geometry was transformed too many times (expected: {n_geom}; got {path_to_geos_counter.call_count}) - the caching is not working.'
    assert len(cgeoaxes._PATH_TRANSFORM_CACHE) == initial_cache_size + n_geom
    fig.clf()
    del path_to_geos_counter
    gc.collect()
    assert len(cgeoaxes._PATH_TRANSFORM_CACHE) == initial_cache_size