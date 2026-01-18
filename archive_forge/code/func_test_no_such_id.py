import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_no_such_id(self):
    error = errors.NoSuchId('atree', 'anid')
    self.assertEqualDiff('The file id "anid" is not present in the tree atree.', str(error))