from unittest import mock
from xml.etree.ElementTree import ParseError
import numpy as np
import pytest
import cartopy.crs as ccrs
from cartopy.tests.conftest import _HAS_PYKDTREE_OR_SCIPY
import cartopy.io.ogc_clients as ogc
from cartopy.io.ogc_clients import _OWSLIB_AVAILABLE
def test_default_projection(self):
    source = ogc.WFSGeometrySource(self.URI, self.typename)
    assert source.default_projection() == self.native_projection