import unittest
from lazr.restfulclient.errors import (
def test_error_for_409(self):
    """Make sure a 409 response code yields Conflict."""
    self.error_for_status(409, Conflict, 'error message')