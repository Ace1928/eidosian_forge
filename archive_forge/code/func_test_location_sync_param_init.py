import param
import pytest
from panel.io.location import Location
from panel.io.state import state
from panel.tests.util import serve_and_request, wait_until
from panel.util import edit_readonly
def test_location_sync_param_init(location):
    p = SyncParameterized()
    location.search = '?integer=1&string=abc'
    location.sync(p)
    assert p.integer == 1
    assert p.string == 'abc'
    location.unsync(p)
    assert location._synced == []
    assert location.search == ''