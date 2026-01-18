import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_corrupt_repository(self):
    repo = self.make_repository('.')
    error = errors.CorruptRepository(repo)
    self.assertEqualDiff('An error has been detected in the repository %s.\nPlease run brz reconcile on this repository.' % repo.controldir.root_transport.base, str(error))