import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.mpl.feature_artist import FeatureArtist, _freeze, _GeomKey
def test_feature_artist_geom_single_path(feature):
    plot_crs = ccrs.PlateCarree(central_longitude=180)
    fig, ax = plt.subplots(subplot_kw={'projection': plot_crs})
    ax.add_feature(feature)
    fig.draw_without_rendering()
    for geom in feature.geometries():
        assert isinstance(cached_paths(geom, plot_crs), mpath.Path)