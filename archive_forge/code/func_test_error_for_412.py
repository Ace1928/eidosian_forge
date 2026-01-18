import unittest
from lazr.restfulclient.errors import (
def test_error_for_412(self):
    """Make sure a 412 response code yields PreconditionFailed."""
    self.error_for_status(412, PreconditionFailed, 'error message')