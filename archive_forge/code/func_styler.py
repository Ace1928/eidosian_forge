import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.mpl.feature_artist import FeatureArtist, _freeze, _GeomKey
def styler(geom):
    if geom == geoms[1]:
        return {'facecolor': 'red'}
    else:
        return {'facecolor': 'blue'}