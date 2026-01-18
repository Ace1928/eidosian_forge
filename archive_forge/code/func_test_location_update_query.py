import param
import pytest
from panel.io.location import Location
from panel.io.state import state
from panel.tests.util import serve_and_request, wait_until
from panel.util import edit_readonly
def test_location_update_query(location):
    location.update_query(a=1)
    assert location.search == '?a=1'
    location.update_query(b='c')
    assert location.search == '?a=1&b=c'