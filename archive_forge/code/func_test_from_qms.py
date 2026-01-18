from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
@pytest.mark.xfail(reason='timeout error', raises=URLError)
def test_from_qms():
    provider = TileProvider.from_qms('OpenStreetMap Standard aka Mapnik')
    assert isinstance(provider, TileProvider)