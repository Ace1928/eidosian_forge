import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_medium_not_connected(self):
    error = errors.MediumNotConnected('a medium')
    self.assertEqualDiff("The medium 'a medium' is not connected.", str(error))