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
def sample_data(shape=(73, 145)):
    """Return ``lons``, ``lats`` and ``data`` of some fake data."""
    nlats, nlons = shape
    lats = np.linspace(-np.pi / 2, np.pi / 2, nlats, dtype=np.longdouble)
    lons = np.linspace(0, 2 * np.pi, nlons, dtype=np.longdouble)
    lons, lats = np.meshgrid(lons, lats)
    wave = 0.75 * np.sin(2 * lats) ** 8 * np.cos(4 * lons)
    mean = 0.5 * np.cos(2 * lats) * (np.sin(2 * lats) ** 2 + 2)
    lats = np.rad2deg(lats)
    lons = np.rad2deg(lons)
    data = wave + mean
    return (lons, lats, data)