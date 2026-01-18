import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_no_smart_medium(self):
    error = errors.NoSmartMedium('a transport')
    self.assertEqualDiff("The transport 'a transport' cannot tunnel the smart protocol.", str(error))