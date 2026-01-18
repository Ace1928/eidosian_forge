import unittest
from lazr.restfulclient.errors import (
def test_no_error_for_5xx(self):
    """Make sure a 5xx response codes yields ServerError."""
    for status in (500, 502, 503, 599):
        self.error_for_status(status, ServerError)