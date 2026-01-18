import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_no_roundtripping(self):
    target_branch = self.make_dummy_builder('dp').get_branch()
    source_tree = self.make_branch_and_tree('dc')
    output, error = self.run_bzr('push -d dc dp', retcode=3)
    self.assertEqual('', output)
    self.assertEqual(error, 'brz: ERROR: It is not possible to losslessly push to dummy. You may want to use --lossy.\n')