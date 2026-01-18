import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_in_process_transport(self):
    error = errors.InProcessTransport('fpp')
    self.assertEqualDiff("The transport 'fpp' is only accessible within this process.", str(error))