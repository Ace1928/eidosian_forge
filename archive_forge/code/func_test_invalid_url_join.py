import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_invalid_url_join(self):
    """Test the formatting of InvalidURLJoin."""
    e = urlutils.InvalidURLJoin('Reason', 'base path', ('args',))
    self.assertEqual("Invalid URL join request: Reason: 'base path' + ('args',)", str(e))