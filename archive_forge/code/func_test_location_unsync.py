import param
import pytest
from panel.io.location import Location
from panel.io.state import state
from panel.tests.util import serve_and_request, wait_until
from panel.util import edit_readonly
def test_location_unsync(location):
    p = SyncParameterized(integer=1, string='abc')
    location.sync(p)
    assert location.search == '?integer=1&string=abc'
    location.unsync(p)
    assert location.search == ''
    location.update_query(integer=2, string='def')
    assert p.integer == 1
    assert p.string == 'abc'
    p.integer = 3
    p.string = 'ghi'
    assert location.search == '?integer=2&string=def'