import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_many_revisions(self):
    tree = self.setup_ab_tree()
    lf = LogCatcher()
    start_rev = revisionspec.RevisionInfo(tree.branch, None, b'1a')
    end_rev = revisionspec.RevisionInfo(tree.branch, None, b'3a')
    log.show_log(tree.branch, lf, verbose=True, start_revision=start_rev, end_revision=end_rev)
    self.assertEqual(3, len(lf.revisions))
    self.assertEqual(None, lf.revisions[0].revno)
    self.assertEqual(b'3a', lf.revisions[0].rev.revision_id)
    self.assertEqual(None, lf.revisions[1].revno)
    self.assertEqual(b'2a', lf.revisions[1].rev.revision_id)
    self.assertEqual('1', lf.revisions[2].revno)