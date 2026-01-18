import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_immortal_pending_deletion_message(self):
    err = errors.ImmortalPendingDeletion('foo')
    self.assertEqual('Unable to delete transform temporary directory foo.  Please examine foo to see if it contains any files you wish to keep, and delete it when you are done.', str(err))