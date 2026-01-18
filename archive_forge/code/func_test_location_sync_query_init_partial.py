import param
import pytest
from panel.io.location import Location
from panel.io.state import state
from panel.tests.util import serve_and_request, wait_until
from panel.util import edit_readonly
def test_location_sync_query_init_partial(location):
    p = SyncParameterized(integer=1, string='abc')
    location.sync(p, ['integer'])
    assert location.search == '?integer=1'
    location.unsync(p)
    assert location._synced == []