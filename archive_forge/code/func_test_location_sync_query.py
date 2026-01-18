import param
import pytest
from panel.io.location import Location
from panel.io.state import state
from panel.tests.util import serve_and_request, wait_until
from panel.util import edit_readonly
def test_location_sync_query(location):
    p = SyncParameterized()
    location.sync(p)
    p.integer = 2
    assert location.search == '?integer=2'
    location.unsync(p)
    assert location._synced == []
    assert location.search == ''