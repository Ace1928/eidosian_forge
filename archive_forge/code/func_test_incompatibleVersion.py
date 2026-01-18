import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_incompatibleVersion(self):
    error = errors.IncompatibleVersion('module', [(4, 5, 6), (7, 8, 9)], (1, 2, 3))
    self.assertEqualDiff('API module is not compatible; one of versions [(4, 5, 6), (7, 8, 9)] is required, but current version is (1, 2, 3).', str(error))