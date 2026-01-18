from urllib.error import URLError
import pytest
import xyzservices.providers as xyz
from xyzservices import Bunch, TileProvider
def test_expect_name_url_attribution():
    msg = 'The attributes `name`, `url`, and `attribution` are required to initialise a `TileProvider`. Please provide values for: '
    with pytest.raises(AttributeError, match=msg + '`name`, `url`, `attribution`'):
        TileProvider({})
    with pytest.raises(AttributeError, match=msg + '`url`, `attribution`'):
        TileProvider({'name': 'myname'})
    with pytest.raises(AttributeError, match=msg + '`attribution`'):
        TileProvider({'url': 'my_url', 'name': 'my_name'})
    with pytest.raises(AttributeError, match=msg + '`attribution`'):
        TileProvider(url='my_url', name='my_name')