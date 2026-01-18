import unittest
from lazr.restfulclient.errors import (
def test_no_error_for_2xx(self):
    """Make sure a 2xx response code yields no error."""
    for status in (200, 201, 209, 299):
        self.error_for_status(status, None)