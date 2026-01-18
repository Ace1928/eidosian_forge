import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
def test_ghost_revisions_have_no_revno(self):
    error = errors.GhostRevisionsHaveNoRevno('target', 'ghost_rev')
    self.assertEqualDiff('Could not determine revno for {target} because its ancestry shows a ghost at {ghost_rev}', str(error))