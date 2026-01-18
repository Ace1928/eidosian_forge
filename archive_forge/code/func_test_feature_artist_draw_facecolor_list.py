import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.mpl.feature_artist import FeatureArtist, _freeze, _GeomKey
@pytest.mark.mpl_image_compare(filename='feature_artist.png')
def test_feature_artist_draw_facecolor_list(feature):
    fig, ax = robinson_map()
    ax.add_feature(feature, facecolor=['red', 'green', 'blue'])
    return fig