import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_invalid_range(self):
    error = errors.InvalidRange('path', 12, 'bad range')
    self.assertEqual('Invalid range access in path at 12: bad range', str(error))