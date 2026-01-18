from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
def test_extra_kwargs_None(self):
    source = ogc.WMSRasterSource(self.URI, self.layer, getmap_extra_kwargs=None)
    assert source.getmap_extra_kwargs == {'transparent': True}