import unittest
from lazr.restfulclient.errors import (
def test_error_for_404(self):
    """Make sure a 404 response code yields Not Found."""
    self.error_for_status(404, NotFound, 'error message')