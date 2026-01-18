import param
import pytest
from panel.io.location import Location
from panel.io.state import state
from panel.tests.util import serve_and_request, wait_until
from panel.util import edit_readonly
def test_location_sync_on_error(location):
    p = SyncParameterized(string='abc')
    changes = []

    def on_error(change):
        changes.append(change)
    location.sync(p, on_error=on_error)
    location.search = '?integer=a&string=abc'
    assert changes == [{'integer': 'a'}]
    location.unsync(p)
    assert location._synced == []
    assert location.search == ''