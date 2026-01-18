import unittest
from lazr.restfulclient.errors import (
def test_error_for_401(self):
    """Make sure a 401 response code yields Unauthorized."""
    self.error_for_status(401, Unauthorized, 'error message')