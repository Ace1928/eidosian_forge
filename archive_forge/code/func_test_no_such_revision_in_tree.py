import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_no_such_revision_in_tree(self):
    error = errors.NoSuchRevisionInTree('atree', 'anid')
    self.assertEqualDiff('The revision id {anid} is not present in the tree atree.', str(error))
    self.assertIsInstance(error, errors.NoSuchRevision)