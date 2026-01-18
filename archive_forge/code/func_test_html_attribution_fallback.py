from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
def test_html_attribution_fallback(basic_provider, html_attr_provider):
    assert basic_provider.html_attribution == basic_provider.attribution
    assert html_attr_provider.html_attribution == '&copy; <a href="https://xyzservices.readthedocs.io">xyzservices</a>'