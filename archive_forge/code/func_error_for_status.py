import unittest
from lazr.restfulclient.errors import (
def error_for_status(self, status, expected_error, content=''):
    """Make sure error_for returns the right HTTPError subclass."""
    request = StubRequest(status)
    error = error_for(request, content)
    if expected_error is None:
        self.assertIsNone(error)
    else:
        self.assertTrue(isinstance(error, expected_error))
        self.assertEqual(content, error.content)