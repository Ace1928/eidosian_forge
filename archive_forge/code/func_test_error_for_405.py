import unittest
from lazr.restfulclient.errors import (
def test_error_for_405(self):
    """Make sure a 405 response code yields MethodNotAllowed."""
    self.error_for_status(405, MethodNotAllowed, 'error message')