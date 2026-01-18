from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
def test_float_resolution(self):
    source = ogc.WMSRasterSource(self.URI, self.layer)
    extent = [-570000, 5100000, 870000, 3500000]
    located_image, = source.fetch_raster(self.projection, extent, [19.5, 39.1])
    img = np.array(located_image.image)
    assert img.shape == (40, 20, 4)